# saens

MIDI のノート列に合わせて、入力オーディオをピッチ近傍で切り貼りする実験的な TypeScript ライブラリです。

## 最小コマンド（npx）

```bash
npx @kongyo2/saens --audio input.wav --out output.wav
```

`元データ.mid` と、その元音源 `2.wav` はパッケージに同梱されます。`--audio` は入力音声で、`2.wav` は入力とは別に内部参照として常に使われます。
