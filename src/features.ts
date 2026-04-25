const frameSeconds = 0.025;
const hopSeconds = 0.01;
const melBandCount = 26;
const mfccCount = 13;
const spectralFeatureCount = 4;
const featureCount = mfccCount + melBandCount + spectralFeatureCount;
const minMelFrequency = 80;
const maxMelFrequency = 8000;
const epsilon = 1e-12;
const maxReferenceSeconds = 0.75;
const maxAlignmentLagFrames = 6;
const dtwBandFrames = 6;

const centroidFeature = mfccCount + melBandCount;
const rolloffFeature = centroidFeature + 1;
const flatnessFeature = rolloffFeature + 1;
const zeroCrossingFeature = flatnessFeature + 1;

const featureWeights = createFeatureWeights();

export interface AudioFeatureIndex {
  sampleRate: number;
  frameSize: number;
  hopSize: number;
  hopSeconds: number;
  frameCount: number;
  featureCount: number;
  features: Float32Array;
  rms: Float32Array;
  logRms: Float32Array;
}

export interface ReferenceFeatureSegment {
  index: AudioFeatureIndex;
  startFrame: number;
  offsets: Int16Array;
  weights: Float32Array;
  meanFeatures: Float32Array;
  meanLogRms: number;
}

interface FftPlan {
  size: number;
  reverse: Int32Array;
  cos: Float64Array;
  sin: Float64Array;
}

interface MelBand {
  start: number;
  center: number;
  end: number;
}

function createFeatureWeights(): Float32Array {
  const weights = new Float32Array(featureCount);
  const mfccWeights = [
    1.2, 1.22, 1.16, 1.08, 1.0, 0.92, 0.84, 0.76, 0.68, 0.6, 0.52, 0.44, 0.36,
  ];
  for (let i = 0; i < mfccCount; i++) {
    weights[i] = mfccWeights[i] ?? 0.4;
  }
  for (let band = 0; band < melBandCount; band++) {
    const normalized = band / Math.max(1, melBandCount - 1);
    const vocalBandBoost =
      normalized > 0.08 && normalized < 0.78
        ? 0.2 * Math.sin((Math.PI * (normalized - 0.08)) / 0.7)
        : 0;
    weights[mfccCount + band] = 0.28 + vocalBandBoost;
  }
  weights[centroidFeature] = 0.72;
  weights[rolloffFeature] = 0.52;
  weights[flatnessFeature] = 0.44;
  weights[zeroCrossingFeature] = 0.42;
  return weights;
}

function nextPowerOfTwo(value: number): number {
  let power = 1;
  while (power < value) power <<= 1;
  return power;
}

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

function createFftPlan(size: number): FftPlan {
  const bits = Math.round(Math.log2(size));
  const reverse = new Int32Array(size);
  for (let i = 0; i < size; i++) {
    let x = i;
    let y = 0;
    for (let b = 0; b < bits; b++) {
      y = (y << 1) | (x & 1);
      x >>= 1;
    }
    reverse[i] = y;
  }

  const cos = new Float64Array(size / 2);
  const sin = new Float64Array(size / 2);
  for (let i = 0; i < size / 2; i++) {
    const angle = (-2 * Math.PI * i) / size;
    cos[i] = Math.cos(angle);
    sin[i] = Math.sin(angle);
  }
  return { size, reverse, cos, sin };
}

function fft(plan: FftPlan, real: Float64Array, imag: Float64Array): void {
  const { size, reverse, cos, sin } = plan;
  for (let i = 0; i < size; i++) {
    const j = reverse[i] ?? 0;
    if (i < j) {
      const tr = real[i] ?? 0;
      const ti = imag[i] ?? 0;
      real[i] = real[j] ?? 0;
      imag[i] = imag[j] ?? 0;
      real[j] = tr;
      imag[j] = ti;
    }
  }

  for (let length = 2; length <= size; length <<= 1) {
    const half = length >> 1;
    const tableStep = size / length;
    for (let start = 0; start < size; start += length) {
      for (let j = 0; j < half; j++) {
        const tableIndex = j * tableStep;
        const wr = cos[tableIndex] ?? 1;
        const wi = sin[tableIndex] ?? 0;
        const even = start + j;
        const odd = even + half;
        const or = real[odd] ?? 0;
        const oi = imag[odd] ?? 0;
        const tr = wr * or - wi * oi;
        const ti = wr * oi + wi * or;
        const er = real[even] ?? 0;
        const ei = imag[even] ?? 0;
        real[odd] = er - tr;
        imag[odd] = ei - ti;
        real[even] = er + tr;
        imag[even] = ei + ti;
      }
    }
  }
}

function createWindow(size: number): Float64Array {
  const window = new Float64Array(size);
  if (size <= 1) {
    window[0] = 1;
    return window;
  }
  for (let i = 0; i < size; i++) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (size - 1));
  }
  return window;
}

function createMelBands(sampleRate: number, fftSize: number): MelBand[] {
  const nyquist = sampleRate / 2;
  const minMel = hzToMel(minMelFrequency);
  const maxMel = hzToMel(Math.min(maxMelFrequency, nyquist * 0.95));
  const points: number[] = [];
  for (let i = 0; i < melBandCount + 2; i++) {
    const mel = minMel + ((maxMel - minMel) * i) / (melBandCount + 1);
    const hz = melToHz(mel);
    points.push(
      Math.max(
        0,
        Math.min(fftSize / 2, Math.floor((hz * fftSize) / sampleRate)),
      ),
    );
  }

  const bands: MelBand[] = [];
  for (let i = 0; i < melBandCount; i++) {
    const start = points[i] ?? 0;
    const center = Math.max(start + 1, points[i + 1] ?? start + 1);
    const end = Math.max(center + 1, points[i + 2] ?? center + 1);
    bands.push({
      start,
      center: Math.min(center, fftSize / 2),
      end: Math.min(end, fftSize / 2),
    });
  }
  return bands;
}

function createDctMatrix(): Float64Array {
  const matrix = new Float64Array(mfccCount * melBandCount);
  const scale = Math.sqrt(2 / melBandCount);
  for (let k = 0; k < mfccCount; k++) {
    const coefficient = k + 1;
    for (let n = 0; n < melBandCount; n++) {
      matrix[k * melBandCount + n] =
        scale * Math.cos((Math.PI * coefficient * (n + 0.5)) / melBandCount);
    }
  }
  return matrix;
}

function secondsToFrame(index: AudioFeatureIndex, seconds: number): number {
  return Math.round(seconds / index.hopSeconds);
}

export function buildAudioFeatureIndex(
  samples: Float32Array,
  sampleRate: number,
): AudioFeatureIndex {
  const frameSize = Math.max(256, Math.floor(frameSeconds * sampleRate));
  const hopSize = Math.max(1, Math.floor(hopSeconds * sampleRate));
  const fftSize = nextPowerOfTwo(frameSize);
  const frameCount = Math.max(
    1,
    Math.floor(Math.max(0, samples.length - frameSize) / hopSize) + 1,
  );
  const features = new Float32Array(frameCount * featureCount);
  const rms = new Float32Array(frameCount);
  const logRms = new Float32Array(frameCount);
  const window = createWindow(frameSize);
  const plan = createFftPlan(fftSize);
  const melBands = createMelBands(sampleRate, fftSize);
  const dct = createDctMatrix();
  const real = new Float64Array(fftSize);
  const imag = new Float64Array(fftSize);
  const power = new Float64Array(fftSize / 2 + 1);
  const logMel = new Float64Array(melBandCount);
  const nyquist = sampleRate / 2;

  for (let frame = 0; frame < frameCount; frame++) {
    real.fill(0);
    imag.fill(0);
    const sampleStart = frame * hopSize;
    let energy = 0;
    let crossings = 0;
    let previous = 0;
    for (let i = 0; i < frameSize; i++) {
      const sample = samples[sampleStart + i] ?? 0;
      const value = sample * (window[i] ?? 1);
      real[i] = value;
      energy += sample * sample;
      if (
        i > 0 &&
        previous * sample < 0 &&
        Math.max(Math.abs(previous), Math.abs(sample)) > 1e-5
      ) {
        crossings++;
      }
      previous = sample;
    }

    const frameRms = Math.sqrt(energy / frameSize);
    rms[frame] = frameRms;
    logRms[frame] = Math.log(frameRms + epsilon);

    fft(plan, real, imag);

    let totalPower = 0;
    let centroidNumerator = 0;
    let logPowerSum = 0;
    for (let bin = 0; bin < power.length; bin++) {
      const binPower =
        (real[bin] ?? 0) * (real[bin] ?? 0) +
        (imag[bin] ?? 0) * (imag[bin] ?? 0);
      power[bin] = binPower;
      totalPower += binPower;
      const frequency = (bin * sampleRate) / fftSize;
      centroidNumerator += frequency * binPower;
      logPowerSum += Math.log(binPower + epsilon);
    }

    const centroid =
      totalPower > epsilon ? centroidNumerator / totalPower / nyquist : 0;
    let cumulative = 0;
    let rolloff = 0;
    const rolloffTarget = totalPower * 0.85;
    for (let bin = 0; bin < power.length; bin++) {
      cumulative += power[bin] ?? 0;
      if (cumulative >= rolloffTarget) {
        rolloff = (bin * sampleRate) / fftSize / nyquist;
        break;
      }
    }
    const flatness =
      totalPower > epsilon
        ? Math.exp(logPowerSum / power.length) / (totalPower / power.length)
        : 0;

    for (let band = 0; band < melBandCount; band++) {
      const melBand = melBands[band];
      if (!melBand) continue;
      let melEnergy = 0;
      for (let bin = melBand.start; bin <= melBand.end; bin++) {
        const binPower = power[bin] ?? 0;
        let weight = 0;
        if (bin < melBand.center) {
          weight = (bin - melBand.start) / (melBand.center - melBand.start);
        } else {
          weight = (melBand.end - bin) / (melBand.end - melBand.center);
        }
        if (weight > 0) melEnergy += binPower * weight;
      }
      logMel[band] = Math.log(melEnergy + epsilon);
    }

    const outOffset = frame * featureCount;
    for (let k = 0; k < mfccCount; k++) {
      let value = 0;
      const dctOffset = k * melBandCount;
      for (let band = 0; band < melBandCount; band++) {
        value += (logMel[band] ?? 0) * (dct[dctOffset + band] ?? 0);
      }
      features[outOffset + k] = value;
    }
    let melMean = 0;
    for (let band = 0; band < melBandCount; band++) {
      melMean += logMel[band] ?? 0;
    }
    melMean /= melBandCount;
    for (let band = 0; band < melBandCount; band++) {
      features[outOffset + mfccCount + band] = (logMel[band] ?? 0) - melMean;
    }
    features[outOffset + centroidFeature] = centroid;
    features[outOffset + rolloffFeature] = rolloff;
    features[outOffset + flatnessFeature] = Math.log(flatness + epsilon);
    features[outOffset + zeroCrossingFeature] = Math.log(
      crossings / Math.max(1, frameSize - 1) + epsilon,
    );
  }

  return {
    sampleRate,
    frameSize,
    hopSize,
    hopSeconds,
    frameCount,
    featureCount,
    features,
    rms,
    logRms,
  };
}

export function normalizeFeatureIndexes(indexes: AudioFeatureIndex[]): void {
  const mean = new Float64Array(featureCount);
  const variance = new Float64Array(featureCount);
  let count = 0;

  for (const index of indexes) {
    let peak = 0;
    for (let frame = 0; frame < index.frameCount; frame++) {
      const value = index.rms[frame] ?? 0;
      if (value > peak) peak = value;
    }
    const threshold = Math.max(peak * 0.01, 1e-6);
    for (let frame = 0; frame < index.frameCount; frame++) {
      if ((index.rms[frame] ?? 0) < threshold) continue;
      const offset = frame * featureCount;
      for (let dim = 0; dim < featureCount; dim++) {
        mean[dim] = (mean[dim] ?? 0) + (index.features[offset + dim] ?? 0);
      }
      count++;
    }
  }

  if (count === 0) return;

  for (let dim = 0; dim < featureCount; dim++) {
    mean[dim] = (mean[dim] ?? 0) / count;
  }

  for (const index of indexes) {
    for (let frame = 0; frame < index.frameCount; frame++) {
      const offset = frame * featureCount;
      for (let dim = 0; dim < featureCount; dim++) {
        const centered = (index.features[offset + dim] ?? 0) - (mean[dim] ?? 0);
        variance[dim] = (variance[dim] ?? 0) + centered * centered;
      }
    }
  }

  for (let dim = 0; dim < featureCount; dim++) {
    variance[dim] = Math.sqrt((variance[dim] ?? 0) / count) || 1;
  }

  for (const index of indexes) {
    for (let frame = 0; frame < index.frameCount; frame++) {
      const offset = frame * featureCount;
      for (let dim = 0; dim < featureCount; dim++) {
        index.features[offset + dim] =
          ((index.features[offset + dim] ?? 0) - (mean[dim] ?? 0)) /
          (variance[dim] ?? 1);
      }
    }
  }
}

function weightedMeanFeatures(
  index: AudioFeatureIndex,
  startFrame: number,
  offsets: Int16Array,
  weights: Float32Array,
): Float32Array {
  const mean = new Float32Array(featureCount);
  let totalWeight = 0;
  for (let i = 0; i < offsets.length; i++) {
    const frame = startFrame + (offsets[i] ?? 0);
    if (frame < 0 || frame >= index.frameCount) continue;
    const weight = weights[i] ?? 0;
    const featureOffset = frame * featureCount;
    for (let dim = 0; dim < featureCount; dim++) {
      mean[dim] =
        (mean[dim] ?? 0) + (index.features[featureOffset + dim] ?? 0) * weight;
    }
    totalWeight += weight;
  }
  if (totalWeight <= 0) return mean;
  for (let dim = 0; dim < featureCount; dim++) {
    mean[dim] = (mean[dim] ?? 0) / totalWeight;
  }
  return mean;
}

function weightedMeanLogRms(
  index: AudioFeatureIndex,
  startFrame: number,
  offsets: Int16Array,
  weights: Float32Array,
): number {
  let total = 0;
  let totalWeight = 0;
  for (let i = 0; i < offsets.length; i++) {
    const frame = startFrame + (offsets[i] ?? 0);
    if (frame < 0 || frame >= index.frameCount) continue;
    const weight = weights[i] ?? 0;
    total += (index.logRms[frame] ?? Math.log(epsilon)) * weight;
    totalWeight += weight;
  }
  return totalWeight > 0 ? total / totalWeight : Math.log(epsilon);
}

export function buildReferenceFeatureSegment(
  index: AudioFeatureIndex,
  startSeconds: number,
  durationSeconds: number,
): ReferenceFeatureSegment | undefined {
  const startFrame = secondsToFrame(index, startSeconds);
  if (startFrame < 0 || startFrame >= index.frameCount) return undefined;

  const durationFrames = Math.max(
    2,
    Math.ceil(
      Math.min(Math.max(durationSeconds, hopSeconds * 2), maxReferenceSeconds) /
        index.hopSeconds,
    ),
  );
  const frameCount = Math.min(durationFrames, index.frameCount - startFrame);
  if (frameCount < 2) return undefined;

  let peak = 0;
  for (let i = 0; i < frameCount; i++) {
    const value = index.rms[startFrame + i] ?? 0;
    if (value > peak) peak = value;
  }
  if (peak <= 0) return undefined;

  const threshold = Math.max(peak * 0.04, 1e-6);
  const rawOffsets: number[] = [];
  const rawWeights: number[] = [];
  for (let i = 0; i < frameCount; i++) {
    const frameRms = index.rms[startFrame + i] ?? 0;
    if (frameRms < threshold) continue;
    const onsetBoost = i * index.hopSeconds < 0.14 ? 1.35 : 1;
    rawOffsets.push(i);
    rawWeights.push(Math.sqrt(frameRms / peak) * onsetBoost);
  }

  if (rawOffsets.length < 2) {
    rawOffsets.length = 0;
    rawWeights.length = 0;
    for (let i = 0; i < frameCount; i++) {
      rawOffsets.push(i);
      rawWeights.push(1);
    }
  }

  const offsets = Int16Array.from(rawOffsets);
  const weights = Float32Array.from(rawWeights);
  const meanFeatures = weightedMeanFeatures(
    index,
    startFrame,
    offsets,
    weights,
  );
  const meanLogRms = weightedMeanLogRms(index, startFrame, offsets, weights);
  return { index, startFrame, offsets, weights, meanFeatures, meanLogRms };
}

function meanFeatureDistance(left: Float32Array, right: Float32Array): number {
  let total = 0;
  let weightTotal = 0;
  for (let dim = 0; dim < featureCount; dim++) {
    const weight = featureWeights[dim] ?? 1;
    total += Math.abs((left[dim] ?? 0) - (right[dim] ?? 0)) * weight;
    weightTotal += weight;
  }
  return weightTotal > 0 ? total / weightTotal : 0;
}

function frameDistance(
  candidate: AudioFeatureIndex,
  candidateFrame: number,
  reference: ReferenceFeatureSegment,
  referenceFrame: number,
): number {
  const candidateOffset = candidateFrame * featureCount;
  const referenceOffset = referenceFrame * featureCount;
  let total = 0;
  let weightTotal = 0;
  for (let dim = 0; dim < featureCount; dim++) {
    const weight = featureWeights[dim] ?? 1;
    total +=
      Math.abs(
        (candidate.features[candidateOffset + dim] ?? 0) -
          (reference.index.features[referenceOffset + dim] ?? 0),
      ) * weight;
    weightTotal += weight;
  }
  return weightTotal > 0 ? total / weightTotal : 0;
}

function deltaDistance(
  candidate: AudioFeatureIndex,
  candidateFrame: number,
  reference: ReferenceFeatureSegment,
  referenceFrame: number,
): number {
  if (candidateFrame <= 0 || referenceFrame <= 0) return 0;
  const candidateOffset = candidateFrame * featureCount;
  const previousCandidateOffset = (candidateFrame - 1) * featureCount;
  const referenceOffset = referenceFrame * featureCount;
  const previousReferenceOffset = (referenceFrame - 1) * featureCount;
  let total = 0;
  let weightTotal = 0;
  for (let dim = 0; dim < mfccCount + melBandCount; dim++) {
    const weight = featureWeights[dim] ?? 1;
    const candidateDelta =
      (candidate.features[candidateOffset + dim] ?? 0) -
      (candidate.features[previousCandidateOffset + dim] ?? 0);
    const referenceDelta =
      (reference.index.features[referenceOffset + dim] ?? 0) -
      (reference.index.features[previousReferenceOffset + dim] ?? 0);
    total += Math.abs(candidateDelta - referenceDelta) * weight;
    weightTotal += weight;
  }
  return weightTotal > 0 ? total / weightTotal : 0;
}

function dtwFeatureDistance(
  candidate: AudioFeatureIndex,
  alignedStartFrame: number,
  reference: ReferenceFeatureSegment,
  candidateMeanLogRms: number,
): number {
  const n = reference.offsets.length;
  if (n < 2) return 0;

  const band = Math.min(n, Math.max(dtwBandFrames, Math.ceil(n * 0.18)));
  let previous = new Float64Array(n + 1);
  let current = new Float64Array(n + 1);
  previous.fill(Infinity);
  previous[0] = 0;

  for (let i = 1; i <= n; i++) {
    current.fill(Infinity);
    const referenceOffset = reference.offsets[i - 1] ?? 0;
    const referenceFrame = reference.startFrame + referenceOffset;
    if (referenceFrame < 0 || referenceFrame >= reference.index.frameCount) {
      const swap = previous;
      previous = current;
      current = swap;
      continue;
    }

    const start = Math.max(1, i - band);
    const end = Math.min(n, i + band);
    for (let j = start; j <= end; j++) {
      const candidateOffset = reference.offsets[j - 1] ?? 0;
      const candidateFrame = alignedStartFrame + candidateOffset;
      if (candidateFrame < 0 || candidateFrame >= candidate.frameCount) {
        continue;
      }

      const featureDistance = frameDistance(
        candidate,
        candidateFrame,
        reference,
        referenceFrame,
      );
      const envelopeDistance = Math.abs(
        (candidate.logRms[candidateFrame] ?? Math.log(epsilon)) -
          candidateMeanLogRms -
          ((reference.index.logRms[referenceFrame] ?? Math.log(epsilon)) -
            reference.meanLogRms),
      );
      const weight =
        ((reference.weights[i - 1] ?? 1) + (reference.weights[j - 1] ?? 1)) / 2;
      const step = Math.min(
        previous[j] ?? Infinity,
        current[j - 1] ?? Infinity,
        previous[j - 1] ?? Infinity,
      );
      current[j] = step + (featureDistance + envelopeDistance * 0.08) * weight;
    }

    const swap = previous;
    previous = current;
    current = swap;
  }

  let weightTotal = 0;
  for (let i = 0; i < reference.weights.length; i++) {
    weightTotal += reference.weights[i] ?? 0;
  }
  const score = previous[n] ?? Infinity;
  return Number.isFinite(score) && weightTotal > 0
    ? score / weightTotal
    : Infinity;
}

export function scoreFeatureSegment(
  candidate: AudioFeatureIndex,
  startSeconds: number,
  reference: ReferenceFeatureSegment,
): number {
  const candidateStartFrame = secondsToFrame(candidate, startSeconds);
  let bestScore = Infinity;

  for (let lag = -maxAlignmentLagFrames; lag <= maxAlignmentLagFrames; lag++) {
    const alignedStartFrame = candidateStartFrame + lag;
    const candidateMeanFeatures = weightedMeanFeatures(
      candidate,
      alignedStartFrame,
      reference.offsets,
      reference.weights,
    );
    const candidateMeanLogRms = weightedMeanLogRms(
      candidate,
      alignedStartFrame,
      reference.offsets,
      reference.weights,
    );

    let frameTotal = 0;
    let envelopeTotal = 0;
    let deltaTotal = 0;
    let used = 0;
    let weightTotal = 0;

    for (let i = 0; i < reference.offsets.length; i++) {
      const offset = reference.offsets[i] ?? 0;
      const candidateFrame = alignedStartFrame + offset;
      const referenceFrame = reference.startFrame + offset;
      if (
        candidateFrame < 0 ||
        candidateFrame >= candidate.frameCount ||
        referenceFrame < 0 ||
        referenceFrame >= reference.index.frameCount
      ) {
        continue;
      }

      const weight = reference.weights[i] ?? 0;
      frameTotal +=
        frameDistance(candidate, candidateFrame, reference, referenceFrame) *
        weight;
      envelopeTotal +=
        Math.abs(
          (candidate.logRms[candidateFrame] ?? Math.log(epsilon)) -
            candidateMeanLogRms -
            ((reference.index.logRms[referenceFrame] ?? Math.log(epsilon)) -
              reference.meanLogRms),
        ) * weight;
      deltaTotal +=
        deltaDistance(candidate, candidateFrame, reference, referenceFrame) *
        weight;
      used++;
      weightTotal += weight;
    }

    if (used === 0 || weightTotal <= 0) continue;
    const coveragePenalty =
      (reference.offsets.length - used) / reference.offsets.length;
    const frameScore = frameTotal / weightTotal;
    const envelopeScore = envelopeTotal / weightTotal;
    const deltaScore = deltaTotal / weightTotal;
    const dtwScore = dtwFeatureDistance(
      candidate,
      alignedStartFrame,
      reference,
      candidateMeanLogRms,
    );
    const meanScore = meanFeatureDistance(
      candidateMeanFeatures,
      reference.meanFeatures,
    );
    const score =
      frameScore * 0.3 +
      dtwScore * 0.28 +
      meanScore * 0.2 +
      deltaScore * 0.13 +
      envelopeScore * 0.07 +
      coveragePenalty * 0.35;
    if (score < bestScore) bestScore = score;
  }

  return bestScore;
}
