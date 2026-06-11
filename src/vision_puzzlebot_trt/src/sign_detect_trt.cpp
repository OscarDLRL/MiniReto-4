
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float32.hpp>
 
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
 
#include <NvInfer.h>
#include <cuda_runtime_api.h>
 
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
 
using std::placeholders::_1;
 
class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};
 
struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};
 
class SignDetectTRT : public rclcpp::Node
{
public:
    SignDetectTRT() : Node("sign_detect_trt")
    {
        this->declare_parameter<std::string>("engine_path", "models/yolov4-tiny-signs_best_fp16.engine");
        this->declare_parameter<std::string>("names_path", "models/obj.names");
        this->declare_parameter<std::string>("input_topic", "/cam/img_raw");
 
        this->declare_parameter<double>("confidence_threshold", 0.05);
        this->declare_parameter<double>("nms_threshold", 0.45);
        this->declare_parameter<bool>("show_window", true);
 
        this->declare_parameter<double>("roi_x1", 0.0);
        this->declare_parameter<double>("roi_y1", 0.0);
        this->declare_parameter<double>("roi_x2", 1.0);
        this->declare_parameter<double>("roi_y2", 1.0);
 
        engine_path_ = this->get_parameter("engine_path").as_string();
        names_path_ = this->get_parameter("names_path").as_string();
        input_topic_ = this->get_parameter("input_topic").as_string();
 
        conf_threshold_ = this->get_parameter("confidence_threshold").as_double();
        nms_threshold_ = this->get_parameter("nms_threshold").as_double();
        show_window_ = this->get_parameter("show_window").as_bool();
 
        roi_x1_ = this->get_parameter("roi_x1").as_double();
        roi_y1_ = this->get_parameter("roi_y1").as_double();
        roi_x2_ = this->get_parameter("roi_x2").as_double();
        roi_y2_ = this->get_parameter("roi_y2").as_double();
 
        load_class_names();
        load_engine();
        
        // Publishers
        sign_pub_ = this->create_publisher<std_msgs::msg::String>("/traffic_sign", 10);
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/traffic_sign/debug_image", 10);
        stop_area_pub_ = this->create_publisher<std_msgs::msg::Float32>("/stop_area", 10);

        // Subscriber
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            input_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&SignDetectTRT::image_callback, this, _1)
        );
 
        // try/catch para no crashear si no hay display ──
        if (show_window_)
        {
            try
            {
                cv::namedWindow("Sign Detect TensorRT", cv::WINDOW_AUTOSIZE);
            }
            catch (const cv::Exception& e)
            {
                RCLCPP_WARN(this->get_logger(),
                    "No se pudo crear ventana (sin display): %s. Desactivando show_window.",
                    e.what());
                show_window_ = false;
            }
        }
        // ─────────────────────────────────────────────────────────
 
        RCLCPP_INFO(this->get_logger(), "Clases cargadas: %zu", class_names_.size());
        RCLCPP_INFO(this->get_logger(), "Input: 1x%dx%dx%d", input_c_, input_h_, input_w_);
        RCLCPP_INFO(this->get_logger(), "Output: 1x%dx%d", output_num_, output_elem_);
        RCLCPP_INFO(this->get_logger(), "sign_detect_trt listo.");
        RCLCPP_INFO(this->get_logger(), "Leyendo imagen desde: %s", input_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Engine: %s", engine_path_.c_str());
    }
 
    ~SignDetectTRT()
    {
        if (device_input_ != nullptr)
        {
            cudaFree(device_input_);
            device_input_ = nullptr;
        }
 
        if (device_output_ != nullptr)
        {
            cudaFree(device_output_);
            device_output_ = nullptr;
        }
 
        if (show_window_)
        {
            cv::destroyAllWindows();
        }
    }
 
private:
    struct TRTDestroy
    {
        template <class T>
        void operator()(T* obj) const
        {
            if (obj)
            {
                obj->destroy();
            }
        }
    };
 
    void load_class_names()
    {
        std::ifstream file(names_path_);
 
        if (!file.is_open())
        {
            throw std::runtime_error("No pude abrir obj.names: " + names_path_);
        }
 
        std::string line;
        while (std::getline(file, line))
        {
            if (!line.empty())
            {
                class_names_.push_back(line);
            }
        }
 
        if (class_names_.empty())
        {
            throw std::runtime_error("obj.names está vacío.");
        }
    }
 
    void load_engine()
    {
        std::ifstream file(engine_path_, std::ios::binary);
 
        if (!file.good())
        {
            throw std::runtime_error("No pude abrir engine: " + engine_path_);
        }
 
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
 
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
 
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_)
        {
            throw std::runtime_error("No se pudo crear TensorRT runtime.");
        }
 
        engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
        if (!engine_)
        {
            throw std::runtime_error("No se pudo deserializar el engine.");
        }
 
        context_.reset(engine_->createExecutionContext());
        if (!context_)
        {
            throw std::runtime_error("No se pudo crear execution context.");
        }
 
        input_index_ = engine_->getBindingIndex("input");
        output_index_ = engine_->getBindingIndex("output");
 
        if (input_index_ < 0 || output_index_ < 0)
        {
            throw std::runtime_error("No encontré bindings llamados input/output.");
        }
 
        nvinfer1::Dims input_dims = engine_->getBindingDimensions(input_index_);
        nvinfer1::Dims output_dims = engine_->getBindingDimensions(output_index_);
 
        input_c_ = input_dims.d[1];
        input_h_ = input_dims.d[2];
        input_w_ = input_dims.d[3];
 
        output_num_ = output_dims.d[1];
        output_elem_ = output_dims.d[2];
 
        input_size_ = static_cast<size_t>(1 * input_c_ * input_h_ * input_w_);
        output_size_ = static_cast<size_t>(1 * output_num_ * output_elem_);
 
        cudaError_t err;
 
        err = cudaMalloc(&device_input_, input_size_ * sizeof(float));
        if (err != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc device_input falló.");
        }
 
        err = cudaMalloc(&device_output_, output_size_ * sizeof(float));
        if (err != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc device_output falló.");
        }
 
        host_output_.resize(output_size_);
    }
 
    cv::Rect get_roi_rect(const cv::Mat& frame)
    {
        int w = frame.cols;
        int h = frame.rows;
 
        int x1 = static_cast<int>(w * roi_x1_);
        int y1 = static_cast<int>(h * roi_y1_);
        int x2 = static_cast<int>(w * roi_x2_);
        int y2 = static_cast<int>(h * roi_y2_);
 
        x1 = std::max(0, std::min(x1, w - 1));
        y1 = std::max(0, std::min(y1, h - 1));
        x2 = std::max(x1 + 1, std::min(x2, w));
        y2 = std::max(y1 + 1, std::min(y2, h));
 
        return cv::Rect(x1, y1, x2 - x1, y2 - y1);
    }
 
    std::vector<float> preprocess(const cv::Mat& img)
    {
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(input_w_, input_h_));
 
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
 
        std::vector<float> input(input_size_);
 
        int channel_size = input_h_ * input_w_;
 
        for (int c = 0; c < input_c_; ++c)
        {
            for (int y = 0; y < input_h_; ++y)
            {
                for (int x = 0; x < input_w_; ++x)
                {
                    input[c * channel_size + y * input_w_ + x] =
                        rgb.at<cv::Vec3f>(y, x)[c];
                }
            }
        }
 
        return input;
    }
 
    std::vector<Detection> postprocess(const cv::Mat& roi, const cv::Rect& roi_rect)
    {
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
 
        int roi_w = roi.cols;
        int roi_h = roi.rows;
 
        float best_debug_score = 0.0f;
        int best_debug_class = -1;
 
        for (int i = 0; i < output_num_; ++i)
        {
            float* pred = host_output_.data() + i * output_elem_;
 
            /*
             Engine output:
             1 x 2535 x 9
 
             Para 5 clases:
             pred[0] = x1
             pred[1] = y1
             pred[2] = x2
             pred[3] = y2
             pred[4] = score clase 0
             pred[5] = score clase 1
             pred[6] = score clase 2
             pred[7] = score clase 3
             pred[8] = score clase 4
 
             No hay objectness separado.
            */
 
            float x1 = pred[0];
            float y1 = pred[1];
            float x2 = pred[2];
            float y2 = pred[3];
 
            int best_class = -1;
            float best_score = 0.0f;
 
            for (int c = 0; c < static_cast<int>(class_names_.size()); ++c)
            {
                float score = pred[4 + c];
 
                if (score > best_score)
                {
                    best_score = score;
                    best_class = c;
                }
            }
 
            float confidence = best_score;
 
            if (confidence > best_debug_score)
            {
                best_debug_score = confidence;
                best_debug_class = best_class;
            }
 
            if (confidence < conf_threshold_)
            {
                continue;
            }
 
            // Si las cajas vienen normalizadas 0..1, convertir a pixeles
            if (x1 <= 1.5f && y1 <= 1.5f && x2 <= 1.5f && y2 <= 1.5f)
            {
                x1 *= roi_w;
                x2 *= roi_w;
                y1 *= roi_h;
                y2 *= roi_h;
            }
 
            int left = static_cast<int>(x1);
            int top = static_cast<int>(y1);
            int right = static_cast<int>(x2);
            int bottom = static_cast<int>(y2);
 
            int width = right - left;
            int height = bottom - top;
 
            if (width <= 2 || height <= 2)
            {
                continue;
            }
 
            left = std::max(0, std::min(left, roi_w - 1));
            top = std::max(0, std::min(top, roi_h - 1));
            width = std::max(1, std::min(width, roi_w - left));
            height = std::max(1, std::min(height, roi_h - top));
 
            cv::Rect box(
                roi_rect.x + left,
                roi_rect.y + top,
                width,
                height
            );
 
            boxes.push_back(box);
            confidences.push_back(confidence);
            class_ids.push_back(best_class);
        }
 
        if (best_debug_class >= 0 &&
            best_debug_class < static_cast<int>(class_names_.size()))
        {
            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Top raw prediction: %s %.3f",
                class_names_[best_debug_class].c_str(),
                best_debug_score
            );
        }
 
        std::vector<int> indices;
        cv::dnn::NMSBoxes(
            boxes,
            confidences,
            conf_threshold_,
            nms_threshold_,
            indices
        );
 
        std::vector<Detection> detections;
 
        for (int idx : indices)
        {
            if (idx < 0 || idx >= static_cast<int>(boxes.size()))
            {
                continue;
            }
 
            Detection det;
            det.class_id = class_ids[idx];
            det.confidence = confidences[idx];
            det.box = boxes[idx];
 
            if (det.class_id >= 0 &&
                det.class_id < static_cast<int>(class_names_.size()))
            {
                detections.push_back(det);
            }
        }
 
        return detections;
    }
 
    std::vector<Detection> infer(const cv::Mat& roi, const cv::Rect& roi_rect)
    {
        std::vector<float> input = preprocess(roi);
 
        cudaMemcpy(
            device_input_,
            input.data(),
            input_size_ * sizeof(float),
            cudaMemcpyHostToDevice
        );
 
        void* bindings[2];
        bindings[input_index_] = device_input_;
        bindings[output_index_] = device_output_;
 
        bool ok = context_->executeV2(bindings);
 
        if (!ok)
        {
            RCLCPP_ERROR(this->get_logger(), "Fallo TensorRT executeV2.");
            return {};
        }
 
        cudaMemcpy(
            host_output_.data(),
            device_output_,
            output_size_ * sizeof(float),
            cudaMemcpyDeviceToHost
        );
 
        return postprocess(roi, roi_rect);
    }
 
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv::Mat frame;
 
        try
        {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error cv_bridge: %s", e.what());
            return;
        }
 
        if (frame.empty())
        {
            return;
        }
 
        cv::Rect roi_rect = get_roi_rect(frame);
        cv::Mat roi = frame(roi_rect).clone();
 
        std::vector<Detection> detections = infer(roi, roi_rect);
 
        cv::Mat debug = frame.clone();
 
        cv::rectangle(debug, roi_rect, cv::Scalar(255, 255, 255), 2);
 
        std::string detected_label = "none";
        float best_conf = 0.0f;
 
        if (!detections.empty())
        {
            auto best = std::max_element(
                detections.begin(),
                detections.end(),
                [](const Detection& a, const Detection& b)
                {
                    return a.box.area() < b.box.area();
                }
            );
 
            if (best != detections.end())
            {
                detected_label = class_names_[best->class_id];
                best_conf = best->confidence;
            }
        }
 
        std_msgs::msg::String out_msg;
        out_msg.data = detected_label;
        sign_pub_->publish(out_msg);

        // Calcula el área más grande de cualquier detección tipo "stop".
        float stop_area = 0.0f;

        for (const auto& det : detections)
        {
            if (class_names_[det.class_id] == "stop")
            {
                float area = static_cast<float>(det.box.width * det.box.height);

                // Si hay varios STOP, se queda con el más grande.
                if (area > stop_area)
                {
                    stop_area = area;
                }
            }
        }

        // Manda el área por ROS2 en el topic /stop_area.
        std_msgs::msg::Float32 area_msg;
        area_msg.data = stop_area;
        stop_area_pub_->publish(area_msg);
 
        for (const auto& det : detections)
        {
            cv::rectangle(debug, det.box, cv::Scalar(0, 255, 0), 2);
 
            std::string label =
                class_names_[det.class_id] + " " + std::to_string(det.confidence).substr(0, 4);
 
            cv::putText(
                debug,
                label,
                cv::Point(det.box.x, std::max(20, det.box.y - 8)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 255, 0),
                2
            );
        }
 
        int panel_w = std::min(620, debug.cols - 20);
        if (panel_w > 100)
        {
            cv::rectangle(debug, cv::Rect(10, 10, panel_w, 80), cv::Scalar(0, 0, 0), -1);
        }
 
        std::string status = "Detectado: " + detected_label;
        if (detected_label != "none")
        {
            status += " (" + std::to_string(best_conf).substr(0, 4) + ")";
        }
 
        cv::putText(
            debug,
            status,
            cv::Point(20, 45),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            detected_label == "none" ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0),
            2
        );
 
        cv::putText(
            debug,
            "TensorRT YOLOv4-tiny",
            cv::Point(20, 75),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 255, 255),
            2
        );
 
        sensor_msgs::msg::Image::SharedPtr debug_msg =
            cv_bridge::CvImage(msg->header, "bgr8", debug).toImageMsg();
 
        debug_pub_->publish(*debug_msg);
 
        if (show_window_)
        {
            cv::imshow("Sign Detect TensorRT", debug);
            cv::waitKey(1);
        }
    }
 
private:
    std::string engine_path_;
    std::string names_path_;
    std::string input_topic_;
 
    double conf_threshold_;
    double nms_threshold_;
    bool show_window_;
 
    double roi_x1_;
    double roi_y1_;
    double roi_x2_;
    double roi_y2_;
 
    TrtLogger logger_;
 
    std::unique_ptr<nvinfer1::IRuntime, TRTDestroy> runtime_{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy> engine_{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroy> context_{nullptr};
 
    int input_index_ = -1;
    int output_index_ = -1;
 
    int input_c_ = 3;
    int input_h_ = 416;
    int input_w_ = 416;
 
    int output_num_ = 2535;
    int output_elem_ = 9;
 
    size_t input_size_ = 0;
    size_t output_size_ = 0;
 
    void* device_input_ = nullptr;
    void* device_output_ = nullptr;
 
    std::vector<float> host_output_;
    std::vector<std::string> class_names_;
    
    // Todos los publishers 
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr sign_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr stop_area_pub_;

    // Subscriber
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
};
 
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
 
    auto node = std::make_shared<SignDetectTRT>();
 
    rclcpp::spin(node);
 
    rclcpp::shutdown();
 
    return 0;
}

