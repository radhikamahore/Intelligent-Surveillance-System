from flask import *
#from main import classify
from clasify import video
from clasify import video_live
import webbrowser
from threading import Timer
app = Flask(__name__)  
 
@app.route('/')  
def upload():  
    return render_template("index.html") 

'''@app.route('/success', methods = ['POST'])
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    #print("IN video_feed")
    return Response(
        classify(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    ) 
'''
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file'] 
        f.save(f.filename)  
        #yield render_template("index.html")        

       	#name1 = video(f.filename)
        return Response(
        video(f.filename),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/success_live', methods = ['POST'])  
def success_live():  
    if request.method == 'POST':  


       	return Response(
        video_live(),
        mimetype='multipart/x-mixed-replace; boundary=frame')        
    
def open_browser():
      webbrowser.open_new('localhost:5000/')  
if __name__ == '__main__':
    Timer(1, open_browser).start();
    app.run(debug = False,threaded=False,host='localhost',port='5000') 
