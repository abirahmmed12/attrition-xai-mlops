from flask import Flask, request, render_template
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
import sys

app = Flask(__name__)


print(" Initializing prediction pipeline...")
pipeline = PredictPipeline()
pipeline.load_models()  
print(" Pipeline ready!")

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        form_data = request.form.to_dict()
        
        for key in form_data:
            if form_data[key].replace('.','',1).replace('-','',1).isdigit():
                form_data[key] = float(form_data[key])
        
        data_obj = CustomData(**form_data)
        pred_df = data_obj.get_data_as_data_frame()
        
        
        prediction, prob, confidence = pipeline.predict(pred_df)
        factors = pipeline.get_influencing_factors(pred_df)

        is_leaving = (prediction == 1)
        risk_pct = round(prob * 100, 1)
        summary = []
        
        if is_leaving:
            for n in factors['negatives']: 
                summary.append(f"🔴 {n}")
            for p in factors['positives']: 
                summary.append(f"🟢 {p}")
            
            msg = f" High Attrition Risk: {risk_pct}%"
            res_class = "danger"
            recom = "Immediate intervention required - Critical retention risk!"
        else:
            for p in factors['positives']: 
                summary.append(f"🟢 {p}")
            for n in factors['negatives']: 
                summary.append(f"🔴 {n}")
            
            retention_pct = round((1 - prob) * 100, 1)
            msg = f" Low Attrition Risk (Retention Likely: {retention_pct}%)"
            res_class = "success"
            recom = "Maintain current engagement strategy."

        return render_template('home.html', 
                               results=msg, 
                               sub_results=" | ".join(summary[:4]),
                               confidence=confidence,
                               recommendation=recom,
                               result_class=res_class)
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
   
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)