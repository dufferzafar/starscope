import { ref } from 'vue'

let pipelineReady: Promise<any> | null = null
let pipe: any | null = null

export function useEmbedder() {
  const loading = ref(false)
  const modelName = 'Xenova/bge-small-en-v1.5'

  async function init() {
    if (pipe) return pipe
    if (!pipelineReady) {
      loading.value = true
      pipelineReady = (async () => {
        const { pipeline } = await import('@xenova/transformers')
        // Quantized model to keep downloads small; cached in IndexedDB
        pipe = await pipeline('feature-extraction', modelName, { quantized: true })
        loading.value = false
        return pipe
      })()
    }
    return pipelineReady
  }

  async function embed(text: string): Promise<Float32Array> {
    const p = await init()
    // Recommended BGE query prefix for retrieval tasks
    const prefixed = text.trim()
      ? `Represent this sentence for searching relevant passages: ${text.trim()}`
      : ''
    const output = await p(prefixed || text, { pooling: 'mean', normalize: true })
    // output is a Tensor-like with .data (Float32Array) and .dims
    const data = output.data as Float32Array
    return data
  }

  return { loading, embed }
} 