export type ImageSize = 'binary' | 'ternary'

export const IMAGE_MODELS = {
  binary: 'prism-ml/bonsai-image-binary-4B-mlx-1bit',
  ternary: 'prism-ml/bonsai-image-ternary-4B-mlx-2bit',
} as const

export const IMAGE_SIZES: Record<
  ImageSize,
  { label: string; model: string; download: string; quality: string; bits: string }
> = {
  binary: {
    label: 'Fast',
    model: IMAGE_MODELS.binary,
    download: '~3.4 GB',
    quality: '1-bit transformer, quicker download',
    bits: '1-bit',
  },
  ternary: {
    label: 'Quality',
    model: IMAGE_MODELS.ternary,
    download: '~3.9 GB',
    quality: '1.58-bit transformer, closer to FLUX.2 Klein',
    bits: '1.58-bit',
  },
}

export const DEFAULT_IMAGE_SIZE: ImageSize = 'binary'
