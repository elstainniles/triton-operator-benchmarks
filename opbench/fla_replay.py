"""Full-storage replay snapshots; no device is initialized by importing this file."""
import hashlib
import math


class ReplayUnsupported(RuntimeError):
    pass


class ReplayMismatch(RuntimeError):
    pass


class RestoreFailure(BaseException):
    """Unrecoverable device state; must cross original tests' except Exception."""
    pass


def support_check(torch, kernel, raw, args, kwargs):
    """Reject launcher state and opaque arguments that cannot be restored."""
    names = list(kernel.src.fn.arg_names)
    if kwargs or len(args) != len(names) + 9:
        raise ReplayUnsupported('Unsupported compiled launcher ABI')
    if any(value is not None for value in args[6:9]):
        raise ReplayUnsupported('Launch metadata/hooks may hold hidden state')
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
        if value != 0:
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
    return dict(supported=True, **hidden, opaque_pointer_arguments=[], launch_hooks=False,
                hidden_scratch_size=0, pointer_signature_verified=True, device_print=False,
                compile_only=False, launcher_flags=flags)


class StorageState:
    """Copy every referenced backing storage, keeping live views/addresses unchanged."""
    def __init__(self, torch, arguments, synchronize, get_npu_format=None):
        self.torch, self.synchronize = torch, synchronize
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
            if tensor.device.type != 'cpu':
                if get_npu_format is None or get_npu_format(tensor) != 2:
                    raise ReplayUnsupported('Only plain NPU ND storage format 2 is supported')
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage._cdata)
            if key in seen:
                self.storages[seen[key]]['argument_indices'].append(index)
                continue
            size = storage.nbytes()
            if size % tensor.element_size():
                raise ReplayUnsupported('Storage byte count is not divisible by its tensor element size')
            # Keep the NPU dtype and storage descriptor unchanged. Reinterpret
            # bytes only after a same-dtype transfer to CPU. The .data view has
            # an independent version counter: restoration must not invalidate
            # an original test's autograd-saved tensors.
            view = tensor.data.as_strided((size // tensor.element_size(),), (1,), 0)
            entry = dict(view=view, bytes=size, device=str(tensor.device),
                         storage_address=storage.data_ptr(), argument_indices=[index],
                         storage_identity=storage._cdata)
            seen[key] = len(self.storages)
            self.storages.append(entry)
            self.pre.append(self.snapshot(view))
        if not self.storages:
            raise ReplayUnsupported('No tensor storage can establish a restorable launch state')
        synchronize()
        self.verify_layouts()

    def layouts(self):
        return [dict(argument_index=index, shape=list(tensor.shape), stride=list(tensor.stride()),
                     storage_offset=tensor.storage_offset(), dtype=str(tensor.dtype),
                     device=str(tensor.device), data_address=tensor.data_ptr(),
                     storage_address=tensor.untyped_storage().data_ptr(),
                     storage_identity=tensor.untyped_storage()._cdata,
                     storage_bytes=tensor.untyped_storage().nbytes())
                for index, tensor in enumerate(self.arguments) if type(tensor) is self.torch.Tensor]

    def verify_layouts(self):
        if self.layouts() != self.original_layouts:
            raise ReplayMismatch('Original tensor address, alias, shape, stride, or storage offset changed')

    def snapshot(self, view):
        return view.to(device='cpu', copy=True).view(self.torch.uint8)

    def capture_post(self):
        self.synchronize()
        self.verify_layouts()
        self.post = [self.snapshot(entry['view']) for entry in self.storages]

    def restore(self, state):
        if state is None:
            raise RestoreFailure('Post state unavailable')
        self.verify_layouts()
        with self.torch.no_grad():
            for entry, snapshot in zip(self.storages, state):
                entry['view'].copy_(snapshot.view(entry['view'].dtype))
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
            actual = self.snapshot(entry['view'])
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
