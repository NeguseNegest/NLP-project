import sys

import transformer_model_demo_fast

sys.modules["transformer_model"] = transformer_model_demo_fast

import gui_app


if __name__ == "__main__":
    print("Demo fast transformer scorer enabled.")
    print(f"N-gram:      {gui_app.AVAILABLE['ngram']}")
    print(f"Transformer: {gui_app.AVAILABLE['transformer']}")
    gui_app.app.run(
        host="0.0.0.0",
        port=gui_app.ARGS.port,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
