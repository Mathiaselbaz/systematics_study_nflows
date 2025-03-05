#include <iostream>



class ProgressBar {
public:
    // Constructor
    ProgressBar(int totalIterations) : totalIterations(totalIterations), currentIteration(0) {}

    // Call this function to update the progress bar at the end of each iteration
    void update() {
        currentIteration++;
        float progress = static_cast<float>(currentIteration) / totalIterations;
        int barWidth = 70;
        int progressWidth = progress * barWidth;

        std::cout << "[";
        for (int i = 0; i < barWidth; i++) {
            if (i < progressWidth) {
                std::cout << "=";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "] " << int(progress * 100.0) << "%\r";
        std::cout.flush();

        if (currentIteration == totalIterations) {
            std::cout << std::endl;
        }
    }
    void reset() {
        currentIteration = 0;
    }

    void setTotalIterations(int totalIterations) {
        this->totalIterations = totalIterations;
    }

private:
    int totalIterations;
    int currentIteration;
};
