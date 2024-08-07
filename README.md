# Vehicle collision alert project


1. Real-time video processing using opencv
2. We load model while processing the video to detect vehicles
3. Distance Measurements: Using Monocular depth estimation to extimate distance accurately because we using only one cam otherwise 
4. Risk Assesments: a. Threshold setting : Define threshold below while vechile is considered too close
                    b. Alert Generation: If vechile too close then generate alert
5. Driver Alert System: Display Visual Alert OR Display Audio Alert
6. Integration and Testing: a. Integrate all components into a cohesive applications
                            b. testing app under various conditions to ensure reliability



* Bonet ROI - defined as a static region at the center bottom of the frame 
* distance calculation - euclidean distance between vechile and the bonnet ROI
* Set using algorithm
