"""Full-storage replay snapshots; no device is initialized by importing this file."""
import hashlib
import math
import ctypes


class ReplayUnsupported(RuntimeError):
    pass


class ReplayMismatch(RuntimeError):
    pass


class RestoreFailure(BaseException):
    """Unrecoverable device state; must cross original tests' except Exception."""
    pass


class AclStorageCopy:
    """Copy raw backing bytes without changing an NPU storage descriptor."""
    def __init__(self):
        self.lib = ctypes.CDLL('libascendcl.so')
        self.copy = self.lib.aclrtMemcpy
        self.copy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                              ctypes.c_size_t, ctypes.c_int]
        self.copy.restype = ctypes.c_int

    def transfer(self, dst, src, nbytes, kind):
        if nbytes == 0:
            return
        rc = self.copy(dst, nbytes, src, nbytes, kind)
        if rc != 0:
            raise ReplayUnsupported('aclrtMemcpy failed: kind=%d bytes=%d rc=%d' % (kind, nbytes, rc))


def check_launch_hooks(args):
    """Accept only no hooks or the pinned runtime's *empty* default chains.

    The current TA exposes HookChain objects even when no hooks are registered.
    Inspect their actual types, identity, and contents; unknown hooks and lazy
    metadata callbacks remain unsupported. Recheck before every timed replay.
    """
    meta, enter, leave = args[6:9]
    if meta is None and enter is None and leave is None:
        return 'none'
    from triton import knobs
    from triton.knobs import HookChain
    from triton.compiler.compiler import LazyDict
    if (type(enter) is not HookChain or type(leave) is not HookChain
            or enter is not knobs.runtime.launch_enter_hook
            or leave is not knobs.runtime.launch_exit_hook
            or type(enter.calls) is not list or type(leave.calls) is not list
            or enter.calls or leave.calls
            or type(meta) is not LazyDict or type(meta.data) is not dict
            or type(meta.extras) is not list or meta.extras):
        raise ReplayUnsupported('Launch metadata/hooks may hold hidden state')
    return 'empty_runtime_hook_chains'


def support_check(torch, kernel, raw, args, kwargs):
    """Reject launcher state and opaque arguments that cannot be restored."""
    names = list(kernel.src.fn.arg_names)
    if kwargs or len(args) != len(names) + 9:
        raise ReplayUnsupported('Unsupported compiled launcher ABI')
    hook_evidence = check_launch_hooks(args)
    spec = getattr(raw, 'launch_spec', None)
    if spec is None:
        raise ReplayUnsupported('Launcher has no inspectable launch_spec')
    if getattr(raw, 'compile_only', False):
        raise ReplayUnsupported('compile_only launcher does not execute a kernel')
    flags = getattr(spec, 'flags', None)
    if not isinstance(flags, int) or flags & (1 << 7):
        # DEVICE_PRINT is bit 7 in this TA launcher.py/launcher ABI; its debug
        # buffer is not among the kernel's Tensor arguments.
        raise ReplayUnsupported('Missing launcher flags or hidden DEVICE_PRINT buffer')
    hidden = {}
    for field in ('workspace_size', 'ordered_locks', 'unordered_locks'):
        value = getattr(spec, field, None)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ReplayUnsupported('Missing launcher state field: ' + field)
        hidden[field] = value
        if field == 'workspace_size' and value > 0:
            # This pinned TA's native launcher allocates workspace per enqueue
            # and releases its owner after execute (launcher_runtime.cpp).
            # Replaying raw(...) retains that normal scratch lifetime; it is
            # not persistent input state that StorageState must restore.
            from triton.backends.ascend.driver import NPULauncher
            if type(raw) is not NPULauncher or not getattr(raw, '_runtime_path', None):
                raise ReplayUnsupported('Workspace lifetime is unknown for this launcher')
        elif value != 0:
            raise ReplayUnsupported('Unrestorable hidden launcher state: %s=%s' % (field, value))
    for field in ('global_scratch_size', 'profile_scratch_size'):
        if getattr(kernel.metadata, field, 0):
            raise ReplayUnsupported('Unrestorable hidden launcher scratch: ' + field)
    signature = {names.index(k) if isinstance(k, str) else k: value
                 for k, value in kernel.src.signature.items()}
    constants = {names.index(k) if isinstance(k, str) else k[0] if isinstance(k, tuple) and len(k) == 1 else k
                 for k in getattr(kernel.src, 'constants', {})}
    for index, value in enumerate(args[9:]):
        declared = signature.get(index)
        pointer = isinstance(declared, str) and declared.startswith('*')
        if pointer:
            if value is not None and type(value) is not torch.Tensor:
                raise ReplayUnsupported('Opaque pointer argument: ' + names[index])
            if value is not None and str(value.device) != 'npu:0':
                raise ReplayUnsupported('Kernel tensor is not on the sole logical npu:0: ' + names[index])
        elif type(value) is torch.Tensor:
            raise ReplayUnsupported('Tensor without a verified pointer signature: ' + names[index])
        elif index in constants or declared == 'constexpr':
            # Constants are compiled into the kernel and not mutable runtime state.
            continue
        elif value is not None and type(value) not in (int, float, bool):
            raise ReplayUnsupported('Opaque runtime argument: ' + names[index])
        elif isinstance(value, float) and not math.isfinite(value):
            raise ReplayUnsupported('Non-finite scalar argument: ' + names[index])
        if declared is None and index not in constants:
            raise ReplayUnsupported('Runtime argument absent from signature: ' + names[index])
    return dict(supported=True, **hidden, opaque_pointer_arguments=[], launch_hooks=False, launch_hook_mode=hook_evidence,
                workspace_lifecycle='per_native_launch' if hidden['workspace_size'] else 'none',
                native_runtime_path=getattr(raw, '_runtime_path', None),
                hidden_scratch_size=0, pointer_signature_verified=True, device_print=False,
                compile_only=False, launcher_flags=flags)


def validate_raw_npu_format(tensor, npu_format):
    """Admit NCDHW only for a complete contiguous five-dimensional base storage.

    The installed Torch-NPU InferFormat.h identifies NCHW, NCDHW and ND as
    unpadded base formats; ACL_FORMAT_NCDHW is 30 in acl_base_rt.h. Keep the
    descriptor and every address unchanged and copy raw bytes with aclrtMemcpy.
    This deliberately excludes NCDHW views, padded allocations and other formats.
    """
    if npu_format in (0, 2):
        return
    if npu_format == 30:
        storage = tensor.untyped_storage()
        if (len(tensor.shape) == 5 and tensor.is_contiguous()
                and tensor.storage_offset() == 0
                and storage.nbytes() == tensor.numel() * tensor.element_size()
                and tensor.data_ptr() == storage.data_ptr()):
            return
        raise ReplayUnsupported('NCDHW format 30 replay requires contiguous full 5D storage without padding or offset')
    raise ReplayUnsupported('NPU storage format %d is not validated for raw replay' % npu_format)


class StorageState:
    """Copy every referenced backing storage, keeping live views/addresses unchanged."""
    def __init__(self, torch, arguments, synchronize, get_npu_format=None):
        self.torch, self.synchronize = torch, synchronize
        self.get_npu_format = get_npu_format
        self.acl = None
        self.arguments = arguments
        self.storages = []
        self.pre, self.post = [], None
        self.checks = 0
        self.original_layouts = self.layouts()
        seen = {}
        synchronize()
        for index, tensor in enumerate(arguments):
            if type(tensor) is not torch.Tensor:
                continue
            if tensor.layout != torch.strided or tensor.is_quantized or tensor.is_meta:
                raise ReplayUnsupported('Non-strided, quantized, or meta tensor is unsupported')
            if tensor.is_conj() or tensor.is_neg():
                raise ReplayUnsupported('Conjugate/negative view requires additional semantic state')
            if tensor.device.type not in ('cpu', 'npu', 'privateuseone'):
                raise ReplayUnsupported('Unsupported tensor device: ' + str(tensor.device))
            npu_format = None
            if tensor.device.type != 'cpu':
                if get_npu_format is None:
                    raise ReplayUnsupported('NPU storage format could not be verified')
                npu_format = int(get_npu_format(tensor))
                validate_raw_npu_format(tensor, npu_format)
                if self.acl is None:
                    self.acl = AclStorageCopy()
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage._cdata)
            if key in seen:
                if self.storages[seen[key]]['npu_format'] != npu_format:
                    raise ReplayUnsupported('Aliased arguments report different NPU formats')
                self.storages[seen[key]]['argument_indices'].append(index)
                continue
            size = storage.nbytes()
            if size % tensor.element_size():
                raise ReplayUnsupported('Storage byte count is not divisible by its tensor element size')
            # Keep the NPU dtype and storage descriptor unchanged. Reinterpret
            # bytes only after a same-dtype transfer to CPU. The .data view has
            # an independent version counter: restoration must not invalidate
            # an original test's autograd-saved tensors.
            view = tensor.data.as_strided((size // tensor.element_size(),), (1,), 0) if npu_format is None else None
            entry = dict(view=view, bytes=size, device=str(tensor.device),
                         storage_address=storage.data_ptr(), argument_indices=[index],
                         storage_identity=storage._cdata, npu_format=npu_format)
            seen[key] = len(self.storages)
            self.storages.append(entry)
            self.pre.append(self.snapshot(entry))
        if not self.storages:
            raise ReplayUnsupported('No tensor storage can establish a restorable launch state')
        synchronize()
        self.verify_layouts()

    def layouts(self):
        return [dict(argument_index=index, shape=list(tensor.shape), stride=list(tensor.stride()),
                     storage_offset=tensor.storage_offset(), dtype=str(tensor.dtype),
                     device=str(tensor.device), data_address=tensor.data_ptr(),
                     npu_format=(int(self.get_npu_format(tensor)) if tensor.device.type != 'cpu'
                                 and self.get_npu_format is not None else None),
                     storage_address=tensor.untyped_storage().data_ptr(),
                     storage_identity=tensor.untyped_storage()._cdata,
                     storage_bytes=tensor.untyped_storage().nbytes())
                for index, tensor in enumerate(self.arguments) if type(tensor) is self.torch.Tensor]

    def verify_layouts(self):
        if self.layouts() != self.original_layouts:
            raise ReplayMismatch('Original tensor address, alias, shape, stride, or storage offset changed')

    def snapshot(self, entry):
        if entry['npu_format'] is None:
            return entry['view'].to(device='cpu', copy=True).view(self.torch.uint8)
        out = self.torch.empty(entry['bytes'], dtype=self.torch.uint8, device='cpu')
        self.acl.transfer(out.data_ptr(), entry['storage_address'], entry['bytes'], 2)
        return out

    def capture_post(self):
        self.synchronize()
        self.verify_layouts()
        self.post = [self.snapshot(entry) for entry in self.storages]

    def restore(self, state):
        if state is None:
            raise RestoreFailure('Post state unavailable')
        self.verify_layouts()
        with self.torch.no_grad():
            for entry, snapshot in zip(self.storages, state):
                if entry['npu_format'] is None:
                    entry['view'].copy_(snapshot.view(entry['view'].dtype))
                else:
                    self.acl.transfer(entry['storage_address'], snapshot.data_ptr(), entry['bytes'], 1)
        self.synchronize()
        self.verify_layouts()

    def restore_pre(self):
        self.restore(self.pre)

    def verify_post(self, count=True):
        self.synchronize()
        self.verify_layouts()
        if self.post is None:
            raise ReplayMismatch('Missing original post-state snapshot')
        for index, (entry, expected) in enumerate(zip(self.storages, self.post)):
            actual = self.snapshot(entry)
            if not self.torch.equal(actual, expected):
                mismatch = int(self.torch.count_nonzero(actual != expected))
                raise ReplayMismatch('Storage %d differs in %d bytes from original post state' % (index, mismatch))
        if count:
            self.checks += 1

    def restore_post(self):
        try:
            self.restore(self.post)
            self.verify_post(count=False)
        except Exception as error:
            raise RestoreFailure('Could not restore original post state') from error

    def evidence(self):
        def digest(tensor):
            return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        return dict(total_bytes=sum(entry['bytes'] for entry in self.storages),
                    argument_index_convention='kernel_argument_zero_based_excluding_9_launcher_control_arguments',
                    argument_layouts=self.original_layouts,
                    storages=[dict(**{key: value for key, value in entry.items() if key != 'view'},
                                   pre_sha256=digest(self.pre[index]),
                                   post_sha256=digest(self.post[index]))
                              for index, entry in enumerate(self.storages)])


def state_restored_replays(storage, launch, profiler_factory, attempt, warmup=5, samples=20):
    """Restore/check outside RECORD; always give the caller its original post-state."""
    attempt.update(original_calls=1, warmup=warmup, warmup_completed=0, samples_completed=0,
                   replay_checks_passed=0, restoration_outside_capture=False,
                   restore_original_post_state=False,
                   preserves_aliases_strides_offsets_addresses=False)
    try:
        for _ in range(warmup):
            storage.restore_pre()
            launch()
            storage.verify_post()
            attempt['warmup_completed'] += 1
        with profiler_factory() as profiler:
            def require_action(expected):
                action = getattr(getattr(profiler, 'current_action', None), 'name', None)
                if action != expected or getattr(profiler, 'stopped', False):
                    raise ReplayMismatch('Expected profiler %s, got %s' % (expected, action))
            for _ in range(samples):
                require_action('NONE')
                storage.restore_pre()
                profiler.step()
                require_action('WARMUP')
                profiler.step()
                require_action('RECORD_AND_SAVE')
                launch()
                storage.synchronize()
                profiler.step()
                require_action('NONE')
                storage.verify_post()
                attempt['samples_completed'] += 1
        attempt.update(restoration_outside_capture=True,
                       preserves_aliases_strides_offsets_addresses=True,
                       replay_equivalence='all_tensor_backing_storage_bytes_exact_each_replay')
    finally:
        attempt['replay_checks_passed'] = storage.checks
        storage.restore_post()
        attempt['restore_original_post_state'] = True
        attempt['storage_state'] = storage.evidence()
