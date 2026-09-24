export type OCRSize = 'tiny' | 'small' | 'medium'

export const OCR_SIZES: Record<
  OCRSize,
  { label: string; params: string; downloadMB: string; detHmean: string; recAcc: string }
> = {
  tiny: { label: 'Tiny', params: '1.5M', downloadMB: '~6 MB', detHmean: '80.6%', recAcc: '73.5%' },
  small: { label: 'Small', params: '7.7M', downloadMB: '~31 MB', detHmean: '84.1%', recAcc: '81.3%' },
  medium: { label: 'Medium', params: '34.5M', downloadMB: '~139 MB', detHmean: '86.2%', recAcc: '83.2%' },
}

export const DEFAULT_OCR_SIZE: OCRSize = 'small'
