export type ClassifySize = 'small' | 'large'

/** Kev checkpoints packaged for the browser (`training/kev-onnx`). Accuracy is Kev's locked test on sources it never trained on. */
export const CLASSIFY_SIZES: Record<
  ClassifySize,
  { label: string; model: string; base: string; params: string; downloadMB: string; outOfDomain: string }
> = {
  small: {
    label: 'Small',
    model: 'midudev/kev-0.8b-ONNX',
    base: 'Kev-0.8B',
    params: '0.8B',
    downloadMB: '~750 MB',
    outOfDomain: '0.70',
  },
  large: {
    label: 'Large',
    model: 'midudev/kev-4b-ONNX',
    base: 'Kev-4B',
    params: '4B',
    downloadMB: '~2.7 GB',
    outOfDomain: '0.84',
  },
}

export const DEFAULT_CLASSIFY_SIZE: ClassifySize = 'small'
