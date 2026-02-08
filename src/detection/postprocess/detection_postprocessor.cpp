#include "detection/postprocess/detection_postprocessor.h"

DetectionPostprocessor::DetectionPostprocessor(
    float threshold, float box_thresh, int max_candidates, float unclip_ratio,
    bool use_dilation, std::string score_mode)
    : threshold(threshold), box_thresh(box_thresh),
      max_candidates(max_candidates), unclip_ratio(unclip_ratio), min_size(3),
      score_mode(std::move(score_mode))
{
    if (use_dilation)
    {
        dilation_kernel = (cv::Mat_<uint8_t>(2, 2) << 1, 1, 1, 1);
    }
}

std::vector<std::array<cv::Point2f, 4>> DetectionPostprocessor::postprocess(
    const cv::Mat &pred_maps, const int64_t original_image_height,
    const int64_t original_image_width) const
{
    std::vector<std::array<cv::Point2f, 4>> boxes;

    if (pred_maps.empty() || (pred_maps.dims != 4 && pred_maps.dims != 2))
    {
        return boxes;
    }

    int map_height = pred_maps.rows;
    int map_width  = pred_maps.cols;

    if (pred_maps.dims == 4)
    {
        map_height = pred_maps.size[2];
        map_width  = pred_maps.size[3];
    }

    cv::Mat pred_map(map_height, map_width, CV_32F);
    if (pred_maps.dims == 4)
    {
        int          idx[]   = {0, 0, 0, 0};
        const float *src_ptr = pred_maps.ptr<float>(idx);
        std::memcpy(pred_map.data, src_ptr,
                    sizeof(float) * map_height * map_width);
    }
    else
    {
        pred_maps.copyTo(pred_map);
    }

    cv::Mat segmentation;
    cv::threshold(pred_map, segmentation, threshold, 1, cv::THRESH_BINARY);
    segmentation.convertTo(segmentation, CV_8U);

    cv::Mat mask = segmentation;
    if (!dilation_kernel.empty())
    {
        cv::dilate(segmentation, mask, dilation_kernel);
    }

    std::vector<BoxResult> boxes_result = boxes_from_bitmap(
        pred_map, mask, static_cast<int>(original_image_width),
        static_cast<int>(original_image_height));

    for (const auto &box : boxes_result)
    {
        if (box.points.size() != 4)
        {
            continue;
        }
        std::array<cv::Point2f, 4> quad = {box.points[0], box.points[1],
                                           box.points[2], box.points[3]};
        boxes.push_back(quad);
    }

    return filter_small_and_clip_boxes(boxes,
                                       static_cast<int>(original_image_height),
                                       static_cast<int>(original_image_width));
}

std::vector<DetectionPostprocessor::BoxResult>
DetectionPostprocessor::boxes_from_bitmap(const cv::Mat &pred,
                                          const cv::Mat &bitmap, int dest_width,
                                          int dest_height) const
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    size_t num_contours =
        std::min(static_cast<size_t>(max_candidates), contours.size());

    std::vector<BoxResult> boxes;
    boxes.reserve(num_contours);

    for (size_t index = 0; index < num_contours; ++index)
    {
        std::vector<cv::Point2f> contour;
        contour.reserve(contours[index].size());
        for (const auto &p : contours[index])
        {
            contour.emplace_back(static_cast<float>(p.x),
                                 static_cast<float>(p.y));
        }

        auto [points, sside] = get_mini_boxes(contour);
        if (sside < min_size)
        {
            continue;
        }

        float score = box_score_slow(pred, contour);
        if (score < box_thresh)
        {
            continue;
        }

        std::vector<std::vector<cv::Point2f>> expanded =
            unclip(points, unclip_ratio);
        if (expanded.size() != 1)
        {
            continue;
        }

        auto [box, box_sside] = get_mini_boxes(expanded[0]);
        if (box_sside < min_size + 2)
        {
            continue;
        }

        for (auto &p : box)
        {
            p.x = std::clamp(std::round(p.x / pred.cols * dest_width), 0.0f,
                             static_cast<float>(dest_width));
            p.y = std::clamp(std::round(p.y / pred.rows * dest_height), 0.0f,
                             static_cast<float>(dest_height));
        }

        boxes.push_back(BoxResult{box, score});
    }

    return boxes;
}

std::vector<std::vector<cv::Point2f>> DetectionPostprocessor::unclip(
    const std::vector<cv::Point2f> &box, float unclip_ratio) const
{
    if (box.size() < 3)
    {
        return {};
    }

    float area      = 0.0f;
    float perimeter = 0.0f;
    for (size_t i = 0; i < box.size(); ++i)
    {
        const cv::Point2f &p0 = box[i];
        const cv::Point2f &p1 = box[(i + 1) % box.size()];
        area += p0.x * p1.y - p1.x * p0.y;
        perimeter += std::hypot(p1.x - p0.x, p1.y - p0.y);
    }
    area = std::fabs(area) * 0.5f;
    if (perimeter <= 0.0f)
    {
        return {box};
    }

    float distance = area * unclip_ratio / perimeter;
    if (distance <= 0.0f)
    {
        return {box};
    }

    bool ccw = false;
    {
        float signed_area = 0.0f;
        for (size_t i = 0; i < box.size(); ++i)
        {
            const cv::Point2f &p0 = box[i];
            const cv::Point2f &p1 = box[(i + 1) % box.size()];
            signed_area += p0.x * p1.y - p1.x * p0.y;
        }
        ccw = signed_area > 0.0f;
    }

    auto normalize = [](const cv::Point2f &v) {
        float len = std::hypot(v.x, v.y);
        if (len == 0.0f)
            return cv::Point2f(0.0f, 0.0f);
        return cv::Point2f(v.x / len, v.y / len);
    };

    std::vector<cv::Point2f> expanded;
    expanded.reserve(box.size());

    for (size_t i = 0; i < box.size(); ++i)
    {
        const cv::Point2f &p0 = box[i];
        const cv::Point2f &p1 = box[(i + 1) % box.size()];
        const cv::Point2f &p2 = box[(i + 2) % box.size()];

        cv::Point2f e1 = p1 - p0;
        cv::Point2f e2 = p2 - p1;

        cv::Point2f n1 =
            ccw ? cv::Point2f(e1.y, -e1.x) : cv::Point2f(-e1.y, e1.x);
        cv::Point2f n2 =
            ccw ? cv::Point2f(e2.y, -e2.x) : cv::Point2f(-e2.y, e2.x);

        n1 = normalize(n1);
        n2 = normalize(n2);

        cv::Point2f p1_shift  = p1 + n1 * distance;
        cv::Point2f p1_shift2 = p1 + n2 * distance;

        cv::Point2f d1 = e1;
        cv::Point2f d2 = e2;

        float det = d1.x * d2.y - d1.y * d2.x;
        if (std::fabs(det) < 1e-6f)
        {
            expanded.push_back(p1_shift);
            continue;
        }

        cv::Point2f r         = p1_shift2 - p1_shift;
        float       t         = (r.x * d2.y - r.y * d2.x) / det;
        cv::Point2f intersect = p1_shift + d1 * t;
        expanded.push_back(intersect);
    }

    return {expanded};
}

std::pair<std::vector<cv::Point2f>, float>
DetectionPostprocessor::get_mini_boxes(
    const std::vector<cv::Point2f> &contour) const
{
    cv::RotatedRect          box = cv::minAreaRect(contour);
    std::vector<cv::Point2f> points(4);
    box.points(points.data());

    std::sort(
        points.begin(), points.end(),
        [](const cv::Point2f &a, const cv::Point2f &b) { return a.x < b.x; });

    int index_1 = 0, index_2 = 1, index_3 = 2, index_4 = 3;
    if (points[1].y > points[0].y)
    {
        index_1 = 0;
        index_4 = 1;
    }
    else
    {
        index_1 = 1;
        index_4 = 0;
    }
    if (points[3].y > points[2].y)
    {
        index_2 = 2;
        index_3 = 3;
    }
    else
    {
        index_2 = 3;
        index_3 = 2;
    }

    std::vector<cv::Point2f> box_points = {points[index_1], points[index_2],
                                           points[index_3], points[index_4]};
    return {box_points, std::min(box.size.width, box.size.height)};
}

float DetectionPostprocessor::box_score_slow(
    const cv::Mat &bitmap, const std::vector<cv::Point2f> &contour) const
{
    int h = bitmap.rows;
    int w = bitmap.cols;

    float xmin = w - 1.0f;
    float xmax = 0.0f;
    float ymin = h - 1.0f;
    float ymax = 0.0f;

    for (const auto &p : contour)
    {
        xmin = std::min(xmin, p.x);
        xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y);
        ymax = std::max(ymax, p.y);
    }

    xmin = std::clamp(xmin, 0.0f, static_cast<float>(w - 1));
    xmax = std::clamp(xmax, 0.0f, static_cast<float>(w - 1));
    ymin = std::clamp(ymin, 0.0f, static_cast<float>(h - 1));
    ymax = std::clamp(ymax, 0.0f, static_cast<float>(h - 1));

    int x0 = static_cast<int>(std::floor(xmin));
    int x1 = static_cast<int>(std::ceil(xmax));
    int y0 = static_cast<int>(std::floor(ymin));
    int y1 = static_cast<int>(std::ceil(ymax));

    cv::Mat mask = cv::Mat::zeros(y1 - y0 + 1, x1 - x0 + 1, CV_8U);

    std::vector<cv::Point> contour_int;
    contour_int.reserve(contour.size());
    for (const auto &p : contour)
    {
        contour_int.emplace_back(static_cast<int>(p.x) - x0,
                                 static_cast<int>(p.y) - y0);
    }

    std::vector<std::vector<cv::Point>> polys = {contour_int};
    cv::fillPoly(mask, polys, cv::Scalar(1));

    cv::Mat    roi      = bitmap(cv::Range(y0, y1 + 1), cv::Range(x0, x1 + 1));
    cv::Scalar mean_val = cv::mean(roi, mask);
    return static_cast<float>(mean_val[0]);
}

std::array<cv::Point2f, 4> DetectionPostprocessor::order_points_clockwise(
    const std::array<cv::Point2f, 4> &pts) const
{
    std::array<cv::Point2f, 4> rect;

    float min_sum     = std::numeric_limits<float>::max();
    float max_sum     = std::numeric_limits<float>::lowest();
    int   min_sum_idx = 0;
    int   max_sum_idx = 0;

    for (int i = 0; i < 4; ++i)
    {
        float s = pts[i].x + pts[i].y;
        if (s < min_sum)
        {
            min_sum     = s;
            min_sum_idx = i;
        }
        if (s > max_sum)
        {
            max_sum     = s;
            max_sum_idx = i;
        }
    }

    rect[0] = pts[min_sum_idx];
    rect[2] = pts[max_sum_idx];

    int idx1 = -1;
    int idx2 = -1;
    for (int i = 0; i < 4; ++i)
    {
        if (i == min_sum_idx || i == max_sum_idx)
        {
            continue;
        }
        if (idx1 == -1)
        {
            idx1 = i;
        }
        else
        {
            idx2 = i;
        }
    }

    float diff1 = pts[idx1].x - pts[idx1].y;
    float diff2 = pts[idx2].x - pts[idx2].y;

    if (diff1 < diff2)
    {
        rect[1] = pts[idx1];
        rect[3] = pts[idx2];
    }
    else
    {
        rect[1] = pts[idx2];
        rect[3] = pts[idx1];
    }

    return rect;
}

std::array<cv::Point2f, 4> DetectionPostprocessor::clip_points_to_image(
    const std::array<cv::Point2f, 4> &points, int img_height,
    int img_width) const
{
    std::array<cv::Point2f, 4> clipped = points;
    for (auto &p : clipped)
    {
        p.x = std::clamp(p.x, 0.0f, static_cast<float>(img_width - 1));
        p.y = std::clamp(p.y, 0.0f, static_cast<float>(img_height - 1));
    }
    return clipped;
}

std::vector<std::array<cv::Point2f, 4>>
DetectionPostprocessor::filter_small_and_clip_boxes(
    const std::vector<std::array<cv::Point2f, 4>> &boxes, int img_height,
    int img_width) const
{
    std::vector<std::array<cv::Point2f, 4>> filtered;
    filtered.reserve(boxes.size());

    for (const auto &box : boxes)
    {
        std::array<cv::Point2f, 4> ordered = order_points_clockwise(box);
        std::array<cv::Point2f, 4> clipped =
            clip_points_to_image(ordered, img_height, img_width);

        float rect_width  = cv::norm(clipped[0] - clipped[1]);
        float rect_height = cv::norm(clipped[0] - clipped[3]);

        if (rect_width <= 3.0f || rect_height <= 3.0f)
        {
            continue;
        }

        filtered.push_back(clipped);
    }

    return filtered;
}
