from flask import Flask, send_from_directory, request, jsonify
import requests
import numpy as np
from scipy.stats import skew, kurtosis, zscore, iqr
from statistics import mean, median, mode, variance, stdev
import torch 

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
@app.route('/')
def index():
        return send_from_directory('Front','index.html')

@app.route('/api/results')
def results():
        model="mistral"
        try:
                data = request.get_json()
                file_content = data.get('content')
                file_content = file_content.split(',')
                file_content_num = [int(x) for x in file_content]

                mean_value = mean(file_content_num)
                median_value = median(file_content_num)
                mode_value = mode(file_content_num)
                variance_value = variance(file_content_num)
                std_dev_value = stdev(file_content_num)
                range_value = max(file_content_num) - min(file_content_num)
                skewness_value = skew(file_content_num)
                kurtosis_value = kurtosis(file_content_num)
                quantiles_value = np.quantile(file_content_num, [0.25, 0.5, 0.75])
                z_scores = zscore(file_content_num).tolist()
                iqr_value = iqr(file_content_num)

                stats = {
                "mean": mean_value,
                "median": median_value,
                "mode": mode_value,
                "variance": variance_value,
                "standard_deviation": std_dev_value,
                "range": range_value,
                "skewness": skewness_value,
                "kurtosis": kurtosis_value,
                "quantiles": quantiles_value.tolist(),
                "z_scores": z_scores,
                "iqr": iqr_value
                }

                prompt = f"Given the data {stats}, find the best machine learning algorithm to predict the next value."

                response = requests.post('http://localhost:11434/api/generate',
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    'temperature': 0.7,
                })
            
                if response.status_code == 200:
                    response_text = response.json()['response']

                    has_relevant_info = "no relevant information" not in response_text.lower()
                    return jsonify({
                    'response': response_text,
                    'confidence': results['average_confidence'],
                    'has_relevant_info': True,
                    'statistics': stats,
                    })
                else:
                    return f"Error generating response: {response.status_code}", False

        except Exception as e:
            print(f"Error connecting to Ollama: {str(e)}")
            return f"Error connecting to Ollama: {str(e)}", False
        
if __name__ == '__main__':
    app.run(debug=False)
