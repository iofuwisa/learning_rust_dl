// can't import wasm lib direct

import * as wasm from "deep_learning";

export function greet() {
    wasm.greet();
}