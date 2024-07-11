from flask import Flask, request, jsonify
from app import moa_app


app = Flask(__name__)

@app.route('/v1/engines/<engine_id>/completions', methods=['POST'])
def completions(engine_id):
    data = request.json
    prompt = data.get('prompt')
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.7)
    rounds = data.get('rounds', 3)
    # = data.get('', )

    # ここにAIモデルを呼び出すロジックを追加
    # 例えば、GPT-3の代わりに他のモデルを呼び出すなど
    response_text = moa_app(prompt, temperature, rounds, max_tokens)
    return jsonify({
        'id': 'cmpl-3evLNHdZjFGfueR2i8k3R9fj',
        'object': 'text_completion',
        'created': 1234567890,
        'model': engine_id,
        'choices': [
            {
                'text': response_text,
                'index': 0,
                'logprobs': None,
                'finish_reason': 'length'
            }
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
