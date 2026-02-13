#include "utils/utils.h"

cv::Mat hwc_to_chw(const cv::Mat &hwc)
{
    CV_Assert(hwc.channels() == 1 || hwc.channels() == 3);

    const int h = hwc.rows;
    const int w = hwc.cols;
    const int c = hwc.channels();

    cv::Mat hwc_f;
    if (hwc.type() != CV_32F && hwc.type() != CV_32FC1 &&
        hwc.type() != CV_32FC3)
        hwc.convertTo(hwc_f, CV_32F);
    else
        hwc_f = hwc;

    int     sizes[] = {c, h, w};
    cv::Mat chw(3, sizes, CV_32F);

    if (c == 1)
    {
        for (int y = 0; y < h; ++y)
        {
            const float *row = hwc_f.ptr<float>(y);
            for (int x = 0; x < w; ++x) { chw.at<float>(0, y, x) = row[x]; }
        }
    }
    else
    {
        for (int y = 0; y < h; ++y)
        {
            const cv::Vec3f *row = hwc_f.ptr<cv::Vec3f>(y);
            for (int x = 0; x < w; ++x)
            {
                const cv::Vec3f &pix   = row[x];
                chw.at<float>(0, y, x) = pix[0];
                chw.at<float>(1, y, x) = pix[1];
                chw.at<float>(2, y, x) = pix[2];
            }
        }
    }

    return chw;
}

std::vector<size_t> sort_box_indices(const std::vector<Box> &boxes,
                                     float                   row_y_threshold)
{
    std::vector<size_t> idx(boxes.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        const auto &A = boxes[a];
        const auto &B = boxes[b];
        if (A[0].y != B[0].y)
            return A[0].y < B[0].y;
        return A[0].x < B[0].x;
    });

    size_t i = 0;
    while (i < idx.size())
    {
        const float row_y = boxes[idx[i]][0].y;
        size_t      j     = i + 1;
        while (j < idx.size() &&
               std::abs(boxes[idx[j]][0].y - row_y) < row_y_threshold)
        {
            ++j;
        }

        std::stable_sort(
            idx.begin() + i, idx.begin() + j,
            [&](size_t a, size_t b) { return boxes[a][0].x < boxes[b][0].x; });

        i = j;
    }

    return idx;
}

cv::Mat image_box_crop(const cv::Mat &image, const Box &points)
{
    float w1 = cv::norm(points[0] - points[1]);
    float w2 = cv::norm(points[2] - points[3]);
    float h1 = cv::norm(points[0] - points[3]);
    float h2 = cv::norm(points[1] - points[2]);

    int crop_w = static_cast<int>(std::round(std::max(w1, w2)));
    int crop_h = static_cast<int>(std::round(std::max(h1, h2)));

    if (crop_w <= 0 || crop_h <= 0)
        return cv::Mat();

    Box targer_quad = {
        cv::Point2f(0.f, 0.f), cv::Point2f(static_cast<float>(crop_w), 0.f),
        cv::Point2f(static_cast<float>(crop_w), static_cast<float>(crop_h)),
        cv::Point2f(0.f, static_cast<float>(crop_h))};

    cv::Mat M = cv::getPerspectiveTransform(points.data(), targer_quad.data());
    cv::Mat dst;
    cv::warpPerspective(image, dst, M, cv::Size(crop_w, crop_h),
                        cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    if (dst.rows > 0 && dst.cols > 0 &&
        static_cast<float>(dst.rows) / dst.cols >= 1.5f)
    {
        cv::rotate(dst, dst, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    return dst;
}

void show_boxes(const cv::Mat &image, const std::vector<Box> &boxes,
                const std::string         &window_name,
                const std::vector<size_t> &sorted_box_indices,
                const std::vector<std::pair<std::string, float>> &cls_results,
                float                                             cls_threshold,
                const std::unordered_map<size_t, std::string>    &rec_texts)
{
    if (boxes.empty())
    {
        return;
    }

    const cv::Scalar color_normal(0, 200, 0);
    const cv::Scalar color_rotated(0, 0, 255);
    const cv::Scalar bg_color(40, 40, 40);
    const bool       has_cls = !cls_results.empty();
    const bool       has_rec = !rec_texts.empty();
    const int        image_w = image.cols;

    // if rec create double-width canvas, otherwise clone
    cv::Mat                          display_image;
    cv::Ptr<cv::freetype::FreeType2> ft2;
    if (has_rec)
    {
        display_image =
            cv::Mat(image.rows, image_w * 2, image.type(), bg_color);
        image.copyTo(display_image(cv::Rect(0, 0, image_w, image.rows)));

        ft2 = cv::freetype::createFreeType2();
        ft2->loadFontData("assets/fonts/DejaVuSansMono.ttf", 0);
    }
    else
    {
        display_image = image.clone();
    }

    std::unordered_map<size_t, bool> rotated_map;
    if (has_cls)
    {
        const size_t count =
            std::min(sorted_box_indices.size(), cls_results.size());
        for (size_t i = 0; i < count; ++i)
        {
            const auto &res = cls_results[i];
            rotated_map[sorted_box_indices[i]] =
                res.first.find("180") != std::string::npos &&
                res.second > cls_threshold;
        }
    }

    for (size_t b = 0; b < boxes.size(); ++b)
    {
        std::vector<cv::Point> points;
        points.reserve(4);
        for (const auto &p : boxes[b])
        {
            points.emplace_back(static_cast<int>(std::round(p.x)),
                                static_cast<int>(std::round(p.y)));
        }

        cv::Scalar color       = color_normal;
        bool       rotated_cls = false;

        if (has_cls)
        {
            auto it = rotated_map.find(b);
            if (it != rotated_map.end() && it->second)
            {
                color       = color_rotated;
                rotated_cls = true;
            }
        }

        cv::polylines(display_image, points, true, color, 2);

        // draw projected box + recognized text on right panel
        if (has_rec)
        {
            auto rec_it = rec_texts.find(b);
            if (rec_it != rec_texts.end())
            {
                cv::Rect bbox = cv::boundingRect(points);
                cv::Rect proj_box(bbox.x + image_w, bbox.y, bbox.width,
                                  bbox.height);
                cv::rectangle(display_image, proj_box, color, 2);

                const std::string &text = rec_it->second;
                if (!text.empty() && bbox.width > 0 && bbox.height > 0)
                {
                    const bool rotate_text =
                        bbox.width < static_cast<int>(1.5 * bbox.height);

                    // dimensions available for text
                    const int text_area_w =
                        rotate_text ? bbox.height - 4 : bbox.width - 4;
                    const int text_area_h =
                        rotate_text ? bbox.width - 4 : bbox.height - 4;

                    // binary search for the largest font height that fits
                    int lo = 6, hi = std::max(text_area_h, 6);
                    int best_font_h = lo;
                    while (lo <= hi)
                    {
                        int      mid      = (lo + hi) / 2;
                        int      baseline = 0;
                        cv::Size sz =
                            ft2->getTextSize(text, mid, -1, &baseline);
                        if (sz.width <= text_area_w && sz.height <= text_area_h)
                        {
                            best_font_h = mid;
                            lo          = mid + 1;
                        }
                        else
                        {
                            hi = mid - 1;
                        }
                    }

                    int      baseline = 0;
                    cv::Size text_size =
                        ft2->getTextSize(text, best_font_h, -1, &baseline);

                    if (rotate_text)
                    {
                        cv::Mat txt_buf = cv::Mat::zeros(
                            text_size.height + baseline + 4,
                            text_size.width + 4, display_image.type());

                        ft2->putText(txt_buf, text,
                                     cv::Point(2, text_size.height + 1),
                                     best_font_h, cv::Scalar(255, 255, 255), -1,
                                     cv::LINE_AA, true);

                        cv::Mat rotated;
                        int     rotate_code = rotated_cls
                                                  ? cv::ROTATE_90_COUNTERCLOCKWISE
                                                  : cv::ROTATE_90_CLOCKWISE;
                        cv::rotate(txt_buf, rotated, rotate_code);

                        cv::Mat resized;
                        cv::resize(rotated, resized,
                                   cv::Size(proj_box.width, proj_box.height), 0,
                                   0, cv::INTER_AREA);

                        cv::Mat mask;
                        cv::cvtColor(resized, mask, cv::COLOR_BGR2GRAY);

                        resized.copyTo(display_image(proj_box), mask);
                    }
                    else
                    {
                        cv::Mat txt_buf = cv::Mat::zeros(
                            text_size.height + baseline + 4,
                            text_size.width + 4, display_image.type());

                        ft2->putText(txt_buf, text,
                                     cv::Point(2, text_size.height + 1),
                                     best_font_h, cv::Scalar(220, 220, 220), -1,
                                     cv::LINE_AA, true);

                        cv::Mat final_text_mat;

                        if (rotated_cls)
                        {
                            cv::rotate(txt_buf, final_text_mat, cv::ROTATE_180);
                        }
                        else
                        {
                            final_text_mat = txt_buf;
                        }

                        cv::Mat resized;
                        cv::resize(final_text_mat, resized,
                                   cv::Size(proj_box.width, proj_box.height), 0,
                                   0, cv::INTER_AREA);

                        cv::Mat mask;
                        cv::cvtColor(resized, mask, cv::COLOR_BGR2GRAY);

                        resized.copyTo(display_image(proj_box), mask);
                    }
                }
            }
        }
    }

    // legend strip below the image
    if (has_cls)
    {
        const int strip_h = 30;
        const int square  = 14;
        const int pad     = 10;

        cv::Mat strip(strip_h, display_image.cols, display_image.type(),
                      cv::Scalar(40, 40, 40));

        int x = pad;
        int y = (strip_h - square) / 2;

        cv::rectangle(strip, cv::Point(x, y), cv::Point(x + square, y + square),
                      color_normal, cv::FILLED);
        x += square + 5;
        cv::putText(strip, "normal", cv::Point(x, y + square - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220),
                    1);
        x += 60;

        cv::rectangle(strip, cv::Point(x, y), cv::Point(x + square, y + square),
                      color_rotated, cv::FILLED);
        x += square + 5;
        cv::putText(strip, "rotated", cv::Point(x, y + square - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220),
                    1);

        cv::vconcat(display_image, strip, display_image);
    }

    cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN,
                          cv::WINDOW_FULLSCREEN);
    cv::Rect win_rect = cv::getWindowImageRect(window_name);
    cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN,
                          cv::WINDOW_NORMAL);

    const int screen_w    = win_rect.width > 0 ? win_rect.width : 1920;
    const int screen_h    = win_rect.height > 0 ? win_rect.height : 1080;
    const int title_bar_h = 30;
    const int usable_h    = screen_h - title_bar_h;

    // scale image to fill screen width while keeping aspect ratio
    double scale_factor = static_cast<double>(screen_w) / display_image.cols;
    if (static_cast<int>(display_image.rows * scale_factor) > usable_h)
    {
        scale_factor = static_cast<double>(usable_h) / display_image.rows;
    }
    int win_w = static_cast<int>(display_image.cols * scale_factor);
    int win_h = static_cast<int>(display_image.rows * scale_factor);

    cv::resizeWindow(window_name, win_w, win_h);
    cv::moveWindow(window_name, 0, 0);
    cv::imshow(window_name, display_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
