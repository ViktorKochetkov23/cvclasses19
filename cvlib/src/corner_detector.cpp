/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray image, 
                                  CV_OUT std::vector<cv::KeyPoint>& keypoints, 
                                  cv::InputArray)
{
    keypoints.clear();

    cv::Mat gray;
    cv::cvtColor(image.getMat(), gray, cv::COLOR_BGR2GRAY);

    const int threshold = 50;

    // Смещения пикселей круга (x, y)
    static const int offset_x[16] = {
         0,  1,  2,  3,
         3,  3,  2,  1,
         0, -1, -2, -3,
        -3, -3, -2, -1
    };

    static const int offset_y[16] = {
        -3, -3, -2, -1,
         0,  1,  2,  3,
         3,  3,  2,  1,
         0, -1, -2, -3
    };

    // Индексы 1,5,9,13
    static const int test_id[4] = {0, 4, 8, 12};

    const int rows = gray.rows;
    const int cols = gray.cols;

    for (int j = 4; j < rows - 4; ++j) {
        const uchar* row_ptr = gray.ptr<uchar>(j);

        for (int i = 4; i < cols - 4; ++i) {
            const uchar center = row_ptr[i];

            // Быстрая проверка 4 пикселей
            int passed = 0;
            for (int k = 0; k < 4; ++k) {
                int idx = test_id[k];
                int px = i + offset_x[idx];
                int py = j + offset_y[idx];

                uchar pix = gray.ptr<uchar>(py)[px];
                if (std::abs(pix - center) > threshold)
                    passed++;
            }
            if (passed < 3)
                continue;

            // Полная проверка 16 пикселей
            bool flag[16];
            for (int k = 0; k < 16; ++k) {
                int px = i + offset_x[k];
                int py = j + offset_y[k];

                uchar pix = gray.ptr<uchar>(py)[px];
                flag[k] = std::abs(pix - center) > threshold;
            }

            // Проверка 9 последовательных
            bool is_corner = false;
            for (int start = 0; start < 16 && !is_corner; ++start) {
                int c = 0;
                for (int t = 0; t < 16; ++t) {
                    int idx = (start + t) & 15; // быстрее чем %
                    if (flag[idx]) {
                        if (++c >= 9) {
                            is_corner = true;
                            break;
                        }
                    } else {
                        c = 0;
                    }
                }
            }

            if (is_corner) {
                keypoints.emplace_back(i, j, 7);
            }
        }
    }
}

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    cv::Mat img = image.getMat();

    // Если изображение цветное, конвертируем в grayscale
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }

    // Параметры дескриптора
    const int desc_length = 256;  // Длина дескриптора в битах (32 байта)
    const int patch_size = 31;    // Размер патча вокруг ключевой точки (нечетный!)
    const int patch_radius = patch_size / 2;

    // Создаем выходной массив для дескрипторов
    // Для 256-битного дескриптора нужно 32 байта на ключевую точку
    descriptors.create(static_cast<int>(keypoints.size()), desc_length / 8, CV_8U);
    cv::Mat desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    // Предварительно сглаживаем изображение для уменьшения шума
    cv::Mat smoothed;
    cv::GaussianBlur(gray, smoothed, cv::Size(5, 5), 2.0, 2.0);

    // Создаем фиксированный набор пар для сравнения (как в BRIEF)
    // Для простоты используем детерминированную схему, не случайную
    std::vector<cv::Point2i> pattern_points_A(desc_length);
    std::vector<cv::Point2i> pattern_points_B(desc_length);

    // Инициализируем псевдослучайный, но детерминированный паттерн
    std::srand(42); // Фиксированное seed для воспроизводимости
    for (int i = 0; i < desc_length; ++i) {
        // Первая точка в круге радиуса patch_radius
        float angle1 = static_cast<float>(std::rand()) / RAND_MAX * 2 * CV_PI;
        float radius1 = static_cast<float>(std::rand()) / RAND_MAX * patch_radius;
        pattern_points_A[i] = cv::Point2i(
            static_cast<int>(radius1 * std::cos(angle1)),
            static_cast<int>(radius1 * std::sin(angle1))
        );

        // Вторая точка в круге радиуса patch_radius
        float angle2 = static_cast<float>(std::rand()) / RAND_MAX * 2 * CV_PI;
        float radius2 = static_cast<float>(std::rand()) / RAND_MAX * patch_radius;
        pattern_points_B[i] = cv::Point2i(
            static_cast<int>(radius2 * std::cos(angle2)),
            static_cast<int>(radius2 * std::sin(angle2))
        );
    }

    // Для каждой ключевой точки вычисляем дескриптор
    for (size_t k = 0; k < keypoints.size(); ++k) {
        const cv::KeyPoint& kp = keypoints[k];
        int x = static_cast<int>(kp.pt.x + 0.5f);
        int y = static_cast<int>(kp.pt.y + 0.5f);

        // Пропускаем точки слишком близко к границе
        if (x - patch_radius < 0 || x + patch_radius >= gray.cols ||
            y - patch_radius < 0 || y + patch_radius >= gray.rows) {
            // Заполняем нулями для точек у границы
            desc_mat.row(static_cast<int>(k)).setTo(0);
            continue;
        }

        // Указатель на текущий дескриптор (256 бит = 32 байта)
        uchar* desc_ptr = desc_mat.ptr(static_cast<int>(k));

        // Вычисляем бинарный дескриптор
        for (int i = 0; i < desc_length; ++i) {
            // Координаты первой точки сравнения
            int x1 = x + pattern_points_A[i].x;
            int y1 = y + pattern_points_A[i].y;

            // Координаты второй точки сравнения
            int x2 = x + pattern_points_B[i].x;
            int y2 = y + pattern_points_B[i].y;

            // Проверяем границы (на всякий случай)
            if (x1 < 0 || x1 >= gray.cols || y1 < 0 || y1 >= gray.rows ||
                x2 < 0 || x2 >= gray.cols || y2 < 0 || y2 >= gray.rows) {
                continue;
            }

            // Получаем значения интенсивности
            uchar intensity1 = smoothed.at<uchar>(y1, x1);
            uchar intensity2 = smoothed.at<uchar>(y2, x2);

            // Бинарный тест: 1 если intensity1 < intensity2, иначе 0
            bool bit_value = (intensity1 < intensity2);

            // Записываем бит в соответствующий байт дескриптора
            if (bit_value) {
                int byte_index = i / 8;
                int bit_index = i % 8;
                desc_ptr[byte_index] |= (1 << bit_index);
            }
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool /*= false*/)
{
    detect(image, keypoints);
    compute(image, keypoints, descriptors);
}
} // namespace cvlib
