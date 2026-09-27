// tests/e2e/helpers.js — pixel-level rendering assertions for the preview
// pipeline. All comparisons are deterministic: the fixtures are real engine
// output, so "the photo must actually be painted" is decidable by machine.

const { PNG } = require('pngjs');

/** Decode a PNG buffer into {width, height, gray: Float32Array} (luma 0-255). */
function decodeGray(pngBuffer) {
  const png = PNG.sync.read(pngBuffer);
  const { width, height, data } = png;
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }
  return { width, height, gray };
}

/** Box-average downscale of a gray raster to cols x rows tiles. */
function tileMeans(gray, width, height, cols, rows) {
  const means = new Float32Array(cols * rows);
  for (let ry = 0; ry < rows; ry++) {
    for (let rx = 0; rx < cols; rx++) {
      const x0 = Math.floor((rx * width) / cols);
      const x1 = Math.floor(((rx + 1) * width) / cols);
      const y0 = Math.floor((ry * height) / rows);
      const y1 = Math.floor(((ry + 1) * height) / rows);
      let sum = 0;
      let n = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          sum += gray[y * width + x];
          n++;
        }
      }
      means[ry * cols + rx] = n ? sum / n : 0;
    }
  }
  return means;
}

/**
 * Black-hole detector: screenshot the photo rect and compare tile-wise
 * against the fixture image. Any tile that is bright in the source but
 * renders near-black on screen means an unpainted region.
 * @returns {{mae: number, blackHoles: Array<{row: number, col: number, srcMean: number, renderedMean: number}>}}
 */
function findBlackHoles(renderedPngBuffer, fixturePngBuffer, cols = 12, rows = 9) {
  const rendered = decodeGray(renderedPngBuffer);
  const source = decodeGray(fixturePngBuffer);
  const rMeans = tileMeans(rendered.gray, rendered.width, rendered.height, cols, rows);
  const sMeans = tileMeans(source.gray, source.width, source.height, cols, rows);
  const blackHoles = [];
  let absSum = 0;
  for (let i = 0; i < cols * rows; i++) {
    absSum += Math.abs(rMeans[i] - sMeans[i]);
    // Bright source content that rendered as near-background black
    if (sMeans[i] > 45 && rMeans[i] < 12) {
      blackHoles.push({ row: Math.floor(i / cols), col: i % cols, srcMean: sMeans[i], renderedMean: rMeans[i] });
    }
  }
  return { mae: absSum / (cols * rows), blackHoles };
}

/** Decode a base64 fixture payload to a PNG buffer. */
function fixturePngBuffer(base64) {
  return Buffer.from(base64, 'base64');
}

/** Luma histogram check: does the buffer contain pixels close to any RGB color? */
function countPixelsNear(pngBuffer, rgb, tolerance = 60) {
  const png = PNG.sync.read(pngBuffer);
  const { width, height, data } = png;
  let count = 0;
  for (let i = 0; i < width * height; i++) {
    const dr = data[i * 4] - rgb[0];
    const dg = data[i * 4 + 1] - rgb[1];
    const db = data[i * 4 + 2] - rgb[2];
    if (dr * dr + dg * dg + db * db < tolerance * tolerance) count++;
  }
  return count;
}

module.exports = { decodeGray, tileMeans, findBlackHoles, fixturePngBuffer, countPixelsNear };
