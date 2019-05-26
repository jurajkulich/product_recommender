from flask import Flask
import recommendations

app = Flask(__name__)


@app.route('/')
def index():
    return "URL for query: /customer_id"

@app.route('/<int:customer_id>')
def show_recommendations(customer_id):
    return str(recommendations.get_recommendation(customer_id)
)

if __name__ == '__main__':
    app.run(debug=True)

