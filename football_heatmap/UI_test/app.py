from flask import Flask, render_template
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/long_task')
def long_task():
    # Simulate a long-running task (e.g., 5 seconds delay)
    time.sleep(5)
    return "Task completed!"

if __name__ == '__main__':
    app.run(debug=True)
