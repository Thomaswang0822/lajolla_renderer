#include "parsers/parse_scene.h"
#include "parallel.h"
#include "image.h"
#include "render.h"
#include "timer.h"
#include <embree4/rtcore.h>
#include <memory>
#include <thread>
#include <vector>
#include <filesystem>

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        std::cout << "[Usage] ./lajolla [-t num_threads] [-o output_file_name] filename.xml" << std::endl;
        return 0;
    }

    int num_threads = std::thread::hardware_concurrency();
    std::string outputfile = "";
    std::string prefix = "";
    std::vector<std::string> filenames;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-t") {
            num_threads = std::stoi(std::string(argv[++i]));
        } else if (std::string(argv[i]) == "-o") {
            prefix = std::string(argv[++i]) + "_";
        } else {
            filenames.push_back(std::string(argv[i]));
            // extract filename
            std::filesystem::path p(filenames.back());
            std::cout << "filename: " << p.stem() << std::endl;
            if (outputfile.compare("") == 0)
                outputfile += p.stem().string() + ".exr";
        }
    }

    RTCDevice embree_device = rtcNewDevice(nullptr);
    parallel_init(num_threads);

    for (const std::string &filename : filenames) {
        Timer timer;
        tick(timer);
        std::cout << "Parsing and constructing scene " << filename << "." << std::endl;
        Scene* scene = parse_scene(filename, embree_device);
        std::cout << "Done. Took " << tick(timer) << " seconds." << std::endl;
        std::cout << "Rendering..." << "using " << num_threads << " threads." << std::endl;
        Image3 img = render(*scene);
        std::cout << "Done. Took " << tick(timer) << " seconds." << std::endl;
        // write file
        std::filesystem::path p(filename);
        outputfile = prefix + p.stem().string() + ".exr";
        imwrite("images/" + outputfile, img);
        std::cout << "Image written to " << "images/" + outputfile << std::endl;
    }

    parallel_cleanup();
    rtcReleaseDevice(embree_device);
    return 0;
}

