# soccer

The environment that I setup in my work place is python 3.8 with pytorch version that supports cuda 12.4.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

After this I installed ultralytics


In the personal pc the env used is foot_analysis. the python version is 3.8
The tracking system that is implemented by the roboflow guy is just make sure that the false detection of the ball will not get into the way. So the guy's tracking will improve only when there are multiple detection of the ball in the same frame (along with the ball).
If in a frame a ball is detected and also a boot (false detection) that is far away, then based on previous centroids of the ball the current ball is detected as actual ball 
Now the task would be to implement 
* Byte track
* Kamlam filter (to detect even when occluded)
* Some other basic tracking algos to better understand the trackings
