/* tslint:disable */
/* eslint-disable */

export class Tokenizer {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Encode many strings at once: returns [n, len_0..len_{n-1}, ids...].
     */
    encodeMany(texts: string[]): Uint32Array;
    encode(text: string): Uint32Array;
    /**
     * Build from the raw GGUF header bytes (vocab + merges never leave WASM).
     */
    constructor(header: Uint8Array);
    /**
     * Raw bytes of a token (may be a partial UTF-8 sequence).
     */
    piece(id: number): Uint8Array;
    readonly vocabSize: number;
}

export function dequant(ty: number, src: Uint8Array): Float32Array;

/**
 * IEEE half floats (little endian) to f32.
 */
export function f16ToF32(src: Uint8Array): Float32Array;

/**
 * Signed z-score of every neuron for the visualization: [layers][ff], positive
 * when the positive pole wins, negative when the negative pole does.
 */
export function heatMap(ext: Float32Array, stats: Float32Array, cols: number, ff: number, std_floor_pos: Float32Array, std_floor_neg: Float32Array): Float32Array;

export function padBlocks(src: Uint8Array, from: number, to: number): Uint8Array;

/**
 * Parse a GGUF header prefix. Returns JSON {dataStart, meta, tensors}; throws
 * "NEED_MORE" when the prefix is too short. String arrays longer than 64
 * entries (vocab, merges, chat template pieces) are summarized by length.
 */
export function parseHeader(bytes: Uint8Array): string;

/**
 * Theme-aware neuron selection.
 *
 * A neuron (pole) is a candidate when this input would enter its top list
 * (activation past the stored entry threshold) and its z-score is at least
 * `z_min`. Candidates are ranked by `z * max(rel, 0)^rel_pow`, where `rel` is
 * the mean cosine similarity between the input's semantic embedding `q` and
 * the embeddings of the neuron's top-8 atlas concepts — i.e. how much the
 * neuron's theme is about this input.
 *
 * stats: [layers][ff][6] = mean_max, std_max, mean_min, std_min, entry_pos, entry_neg
 * top8:  [layers][ff][16] concept ids (8 positive, then 8 negative)
 * emb:   [concepts][dim] unit vectors; q: [dim] unit vector
 * Returns k rows of [layer, neuron, pole, score, z, act, rel].
 */
export function selectNeurons(ext: Float32Array, stats: Float32Array, ff: number, floor_pos: Float32Array, floor_neg: Float32Array, top8: Uint16Array, emb: Float32Array, q: Float32Array, min_layer: number, z_min: number, require_entry: boolean, rel_pow: number, k: number): Float32Array;

/**
 * Semantic embedding of an input: mean residual over the masked tokens,
 * centered by `mu`, projected by `proj` ([d][k]) and L2-normalized.
 */
export function semEmbed(resid: Float32Array, d: number, mask: Uint8Array, mu: Float32Array, proj: Float32Array, k: number): Float32Array;

/**
 * acts: [layers][tokens][ff]. Returns [layers][ff][2] = (max, min) over the
 * tokens where mask[t] != 0.
 */
export function spanExtrema(acts: Float32Array, layers: number, tokens: number, ff: number, mask: Uint8Array): Float32Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_tokenizer_free: (a: number, b: number) => void;
    readonly dequant: (a: number, b: number, c: number) => [number, number, number, number];
    readonly f16ToF32: (a: number, b: number) => [number, number];
    readonly heatMap: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number];
    readonly padBlocks: (a: number, b: number, c: number, d: number) => [number, number];
    readonly parseHeader: (a: number, b: number) => [number, number, number, number];
    readonly selectNeurons: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number) => [number, number];
    readonly semEmbed: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number];
    readonly spanExtrema: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly tokenizer_encode: (a: number, b: number, c: number) => [number, number];
    readonly tokenizer_encodeMany: (a: number, b: number, c: number) => [number, number];
    readonly tokenizer_new: (a: number, b: number) => [number, number, number];
    readonly tokenizer_piece: (a: number, b: number) => [number, number];
    readonly tokenizer_vocabSize: (a: number) => number;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
