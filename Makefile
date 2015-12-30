
# Common includes and paths for opencv
CFLAGS	  = -I/usr/include/opencv -std=c++11 
LIBS	  = -lopencv_core -lopencv_highgui -lopencv_calib3d -lopencv_imgproc -lopencv_features2d -lopencv_gpu
NVCC	  = nvcc -ccbin
################################################################################

# Target rules
all: cpu


cpu: ComputeDisparity-cpu

opencv-stereo-util.o:opencv-stereo-util.cpp
	g++ $(CFLAGS) $(INCLUDES) -o $@ -c $< $(LIBS)

pushbroom-stereo.o:pushbroom-stereo.cpp
	g++ $(CFLAGS) $(INCLUDES) -o $@ -c $< $(LIBS)

ComputeDisparityMIT.o:ComputeDisparityMIT.cpp
	g++ $(CFLAGS) $(INCLUDES) -o $@ -c $< $(LIBS)

ComputeDisparity-cpu: opencv-stereo-util.o pushbroom-stereo.o ComputeDisparityMIT.o
	g++ -o $@ $+ $(LIBS)


gpu: ComputeDisparity-gpu

getSADCUDA.o:getSADCUDA.cu
	$(NVCC) g++ $(CFLAGS) $(INCLUDES) -o $@ -c $<

pushbroom-stereo-gpu.o:pushbroom-stereo.cpp
	$(NVCC) g++ $(CFLAGS) $(INCLUDES) -o $@ -c $< $(LIBS)

ComputeDisparity-gpu: getSADCUDA.o opencv-stereo-util.o pushbroom-stereo-gpu.o ComputeDisparityMIT.o
	$(NVCC) g++ -o $@ $+ $(LIBS)


run: build
	./ComputeDisparity

clean:
	rm -f ComputeDisparity getSADCUDA.o opencv-stereo-util.o pushbroom-stereo.o ComputeDisparityMIT.o ComputeDisparity-gpu ComputeDisparity-cpu

clobber: clean
