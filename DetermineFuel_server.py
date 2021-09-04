import DetermineFuel # Import the python file containing the ML model
from flask import Flask, request, render_template,jsonify # Import flask libraries

# Initialize the flask class and specify the templates directory
app = Flask(__name__,template_folder="templates")

# Default route set as 'home'
@app.route('/')
def home():
    return render_template('home_fuel.html') # Render home.html

# Route 'classify' accepts GET request
@app.route('/classify',methods=['POST','GET'])
def classify_type():
    try:
        price = request.args.get('price') # Get parameters for sepal length
        kms_driven = request.args.get('kms') # Get parameters for sepal width
        mileage = request.args.get('mileage') # Get parameters for petal length
        engine = request.args.get('engine') # Get parameters for petal width
        max_power = request.args.get('power') # Get parameters for max power
        
        # Get the output from the classification model
        variety = DetermineFuel.classify(price, kms_driven, mileage, engine, max_power)
        
        # Render the output in new HTML page
        return render_template('output_fuel.html', variety=variety)
    except:
        return 'Error'

# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=False)        