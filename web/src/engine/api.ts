// Contract between the UI (main thread) and the inference engine (worker).

export interface ModelSpec {
  id: string;            // 'qwen3.5-4b' | 'qwen3.5-0.8b'
  name: string;          // display name
  url: string;           // GGUF url (Hugging Face)
  bytes: number;         // download size
  atlas: string;         // base path of the precomputed concept atlas
}

export interface ModelInfo {
  id: string;
  name: string;
  layers: number;                        // 32 for 4B
  neurons: number;                       // MLP neurons per layer (9216 for 4B)
  layerTypes: ('linear' | 'full')[];     // Gated DeltaNet vs full attention per layer
  adapter: string;                       // GPU description
}

export type LoadPhase = 'checking' | 'download' | 'cached' | 'upload' | 'atlas' | 'ready';

export interface LoadProgress {
  phase: LoadPhase;
  loaded: number;        // bytes
  total: number;         // bytes
  layersReady: number;   // how many transformer layers are fully on the GPU (for the build-up animation)
  message?: string;
}

export interface TokenInfo {
  text: string;          // decoded token text (may start with a space)
  id: number;
}

export interface NeuronHit {
  layer: number;
  index: number;
  pole: 1 | -1;          // which side of the neuron fired: +1 positive, -1 negative
  z: number;             // how many baseline std devs past normal
  act: number;           // raw activation (max for +, min for −) over the input
  perToken: number[];    // activation at each input token (same order as Analysis.tokens), then at the end of the message
  match?: number;        // 0..1 similarity between the input and the neuron's top concepts
  preview?: string[];    // the neuron's strongest atlas concepts for this pole
}

export interface Analysis {
  text: string;
  tokens: TokenInfo[];   // the tokens of the user's text
  at: number | null;     // the token the neurons were read at (tokens.length = end of the message), null = the end of the text
  reading: number[];     // every position that was read (same indexing as `at`)
  top: NeuronHit[];      // best neurons, strongest first (≈24)
  heat: Float32Array;    // [layers * neurons] signed z-score of every neuron for this input (+ = positive pole)
  ms: number;            // forward pass time
}

export interface AtlasConcept {
  text: string;
  act: number;           // activation measured offline on the concept atlas
  category?: string;
}

export interface NeuronAtlas {
  layer: number;
  index: number;
  positive: AtlasConcept[];  // concepts that drive the neuron up, strongest first
  negative: AtlasConcept[];  // concepts that drive it down (most negative first)
  baseline: { meanMax: number; stdMax: number; meanMin: number; stdMin: number };
}

export interface ProbeResult {
  text: string;
  act: number;           // live activation (pole-aware: max for +, min for −)
  z: number;             // z-score against the baseline for that pole
}

export interface Engine {
  init(): Promise<{ webgpu: boolean; adapter?: string; reason?: string }>;
  isCached(model: ModelSpec): Promise<boolean>;
  load(model: ModelSpec, onProgress: (p: LoadProgress) => void): Promise<ModelInfo>;
  analyze(text: string, at?: number | null): Promise<Analysis>;
  atlas(layer: number, index: number): Promise<NeuronAtlas>;
  probe(layer: number, index: number, pole: 1 | -1, texts: string[], onResult?: (r: ProbeResult) => void): Promise<ProbeResult[]>;
}

export const MODELS: ModelSpec[] = [
  {
    id: 'qwen3.5-4b',
    name: 'Qwen3.5-4B',
    // pinned to the exact commit the concept atlas was computed from
    url: 'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf',
    bytes: 2740937888,
    atlas: 'atlas/qwen3.5-4b',
  },
  {
    id: 'qwen3.5-0.8b',
    name: 'Qwen3.5-0.8B',
    url: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q4_K_M.gguf',
    bytes: 532517120,
    atlas: 'atlas/qwen3.5-0.8b',
  },
];
