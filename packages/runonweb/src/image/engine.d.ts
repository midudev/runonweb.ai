export type BonsaiLoadProgress = {
  phase?: string
  component?: string
  loaded?: number
  total?: number
  fromCache?: boolean
  inFlight?: boolean
}

export type BonsaiGenerateOptions = {
  prompt: string
  width?: number
  height?: number
  numInferenceSteps?: number
  seed?: number
  signal?: AbortSignal
  callbackOnStepEnd?: (
    pipeline: unknown,
    step: number,
    timestep: number,
    extras: { latents: unknown }
  ) => void | Promise<void>
}

export type BonsaiLoadOptions = {
  onProgress?: (status: BonsaiLoadProgress) => void
  cache?: boolean
  cacheName?: string
  force?: boolean
  signal?: AbortSignal
  fetch?: typeof fetch
  repoId?: string
  revision?: string
}

export class BonsaiImageResult {
  bytes: Uint8Array
  width: number
  height: number
  prompt: string
  seed: number
  toBlob(): Blob
  toDataURL(): Promise<string>
  toImageBitmap(): Promise<ImageBitmap>
}

export class BonsaiImagePipeline {
  static isSupported(): Promise<boolean>
  static from_pretrained(
    modelId?: string | null,
    options?: BonsaiLoadOptions
  ): Promise<BonsaiImagePipeline>
  static clearCache(cacheName?: string | null): Promise<void>
  generate(options: BonsaiGenerateOptions): Promise<BonsaiImageResult>
  destroy(): Promise<void>
  getModelName(): string
}
