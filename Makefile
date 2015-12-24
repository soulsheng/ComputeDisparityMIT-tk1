
# Common includes and paths for opencv
CFLAGS	  = -I/usr/include/opencv -std=c++11 
LIBS	  = -lopencv_core -lopencv_highgui -lopencv_calib3d -lopencv_imgproc -lopencv_features2d

################################################################################

# Target rules
all: cpu


cpu: ComputeDisparity

opencv-stereo-util.o:opencv-stereo-util.cpp
	g++ $(CFLAGS) $(INCLUDES) -o $@ -c $< $(LIBS)

pushbroom-stereo.o:pushbroom-stereo.cpp
	g++ $(CFLAGS) $(INCLUDES) -o $@ -c $< $(LIBS)

ComputeDisparityMIT.o:ComputeDisparityMIT.cpp
	g++ $(CFLAGS) $(INCLUDES) -o $@ -c $< $(LIBS)

ComputeDisparity: opencv-stereo-util.o pushbroom-stereo.o ComputeDisparityMIT.o
	g++ -o $@ $+ $(LIBS)

run: build
	./ComputeDisparity

clean:
	rm -f ComputeDisparity getSADCUDA.o opencv-stereo-util.o pushbroom-stereo.o ComputeDisparityMIT.o 

clobber: clean
