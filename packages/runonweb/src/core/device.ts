export type Device = 'webgpu' | 'wasm' | 'auto'
export type ResolvedDevice = 'webgpu' | 'wasm'

let webgpuCache: boolean | null = null

/**
 * Detect whether WebGPU is available in this environment.
 * Result is cached after the first successful check.
 */
export async function isWebGPUAvailable(): Promise<boolean> {
  if (webgpuCache !== null) return webgpuCache

  try {
    const gpu = (globalThis as { navigator?: { gpu?: GPU } }).navigator?.gpu
    if (!gpu) {
      webgpuCache = false
      return false
    }
    const adapter = await gpu.requestAdapter()
    webgpuCache = adapter != null
    return webgpuCache
  } catch {
    webgpuCache = false
    return false
  }
}

/**
 * Resolve `auto` to `webgpu` when available, otherwise `wasm`.
 */
export async function resolveDevice(device: Device = 'auto'): Promise<ResolvedDevice> {
  if (device === 'webgpu' || device === 'wasm') return device
  return (await isWebGPUAvailable()) ? 'webgpu' : 'wasm'
}
