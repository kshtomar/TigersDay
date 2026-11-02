// Runs MCTS off the main thread so the page never freezes while searching
// ONNX Runtime Web (WebAssembly CPU Inference)
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
importScripts(ORT + 'ort.min.js', 'state.js', 'engine.js', 'mcts.js');
// workers have no script tag for ort to find its wasm from
ort.env.wasm.wasmPaths = ORT;

const PROGRESS_MS = 100;
const model = new TDMCTS.ONNXModelWrapper(new URLSearchParams(location.search).get('model'));
const mcts = new TDMCTS.MCTS(model);
model.init();

const report = (done, bestMove) => postMessage({
  done,
  bestMove,
  score: mcts.root.eval,
  visits: mcts.root.visit_count,
  lines: mcts.getTopCandidateLines(3)
});

onmessage = async ({ data }) => {
  mcts.simulations = data.sims;
  let last = performance.now();
  const { bestMove } = await mcts.findMove(Object.assign(new GameState(), data.state), 0.0, () => {
    if (performance.now() - last < PROGRESS_MS) return;
    last = performance.now();
    report(false);
  });
  report(true, bestMove);
};
