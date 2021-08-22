// wasm lib only allowed dynamic import.
import('./deep_learning_wrapper.js')
      .then(module => {
        module.greet();

      })
      .catch(err => {
        console.log("import module error.");
      });