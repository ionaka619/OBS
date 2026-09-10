/* OBS Face Sticker - GPL-2.0-or-later */
#include <obs-module.h>
#include <graphics/image-file.h>
#include <util/platform.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "plugin-support.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

namespace {
struct Placement { float x, y, width; };
struct Detection {
	float x = 0, y = 0, w = 0, h = 0, angle = 0;
	bool found = false;
	uint64_t timestamp_ns = 0;
};
struct FaceStickerFilter {
	obs_source_t *source = nullptr;
	gs_effect_t *effect = nullptr;
	gs_image_file_t image = {};
	bool image_loaded = false;
	std::string sticker = "glasses", custom_path;
	float scale = 1, offset_x = 0, offset_y = 0, smoothing = .82f;
	int detect_every = 3, frame_counter = 0;
	uint32_t frame_width = 0, frame_height = 0;
	cv::CascadeClassifier face_cascade, eye_cascade;
	std::thread worker;
	std::mutex work_mutex, result_mutex;
	std::condition_variable work_cv;
	cv::Mat pending_gray;
	bool work_pending = false, worker_busy = false, stopping = false;
	Detection target, current;
};

static const char *sticker_ids[] = {"glasses", "heart-eyes", "crown", "cat-ears",
	"bunny-ears", "moustache", "blush", "halo"};

static Placement placement_for(const std::string &id)
{
	if (id == "crown") return {.50f, -.02f, 1.25f};
	if (id == "cat-ears") return {.50f, .02f, 1.30f};
	if (id == "bunny-ears") return {.50f, -.12f, 1.18f};
	if (id == "moustache") return {.50f, .68f, .76f};
	if (id == "blush") return {.50f, .61f, 1.12f};
	if (id == "halo") return {.50f, -.14f, 1.20f};
	if (id == "heart-eyes") return {.50f, .43f, 1.10f};
	return {.50f, .43f, 1.18f};
}

static bool frame_to_gray(const obs_source_frame *frame, cv::Mat &gray)
{
	if (!frame || !frame->data[0] || !frame->width || !frame->height) return false;
	const int w = (int)frame->width, h = (int)frame->height;
	try {
		switch (frame->format) {
		case VIDEO_FORMAT_BGRA:
		case VIDEO_FORMAT_BGRX: {
			cv::Mat p(h, w, CV_8UC4, frame->data[0], frame->linesize[0]);
			cv::cvtColor(p, gray, cv::COLOR_BGRA2GRAY); break;
		}
		case VIDEO_FORMAT_RGBA: {
			cv::Mat p(h, w, CV_8UC4, frame->data[0], frame->linesize[0]);
			cv::cvtColor(p, gray, cv::COLOR_RGBA2GRAY); break;
		}
		case VIDEO_FORMAT_BGR3: {
			cv::Mat p(h, w, CV_8UC3, frame->data[0], frame->linesize[0]);
			cv::cvtColor(p, gray, cv::COLOR_BGR2GRAY); break;
		}
		case VIDEO_FORMAT_I420:
		case VIDEO_FORMAT_NV12:
		case VIDEO_FORMAT_I444:
		case VIDEO_FORMAT_Y800: {
			cv::Mat y(h, w, CV_8UC1, frame->data[0], frame->linesize[0]);
			gray = y.clone(); break;
		}
		default: return false;
		}
	} catch (const cv::Exception &) { return false; }
	return !gray.empty();
}

static void tracking_worker(FaceStickerFilter *f)
{
	for (;;) {
		cv::Mat gray;
		{
			std::unique_lock<std::mutex> lock(f->work_mutex);
			f->work_cv.wait(lock, [&] { return f->stopping || f->work_pending; });
			if (f->stopping) return;
			gray = std::move(f->pending_gray);
			f->work_pending = false; f->worker_busy = true;
		}
		Detection out; out.timestamp_ns = os_gettime_ns();
		try {
			cv::Mat small;
			double ratio = gray.cols > 480 ? 480.0 / gray.cols : 1.0;
			if (ratio < 1) cv::resize(gray, small, {}, ratio, ratio, cv::INTER_AREA);
			else small = gray;
			cv::equalizeHist(small, small);
			std::vector<cv::Rect> faces;
			f->face_cascade.detectMultiScale(small, faces, 1.1, 4, 0, {48, 48});
			if (!faces.empty()) {
				auto face = *std::max_element(faces.begin(), faces.end(),
					[](auto &a, auto &b) { return a.area() < b.area(); });
				out = {(float)face.x / small.cols, (float)face.y / small.rows,
					(float)face.width / small.cols, (float)face.height / small.rows,
					0, true, out.timestamp_ns};
				if (!f->eye_cascade.empty()) {
					std::vector<cv::Rect> eyes;
					f->eye_cascade.detectMultiScale(small(face), eyes, 1.1, 3, 0, {12, 12});
					if (eyes.size() >= 2) {
						std::sort(eyes.begin(), eyes.end(), [](auto &a, auto &b) { return a.area() > b.area(); });
						cv::Point2f a(eyes[0].x + eyes[0].width / 2.0f, eyes[0].y + eyes[0].height / 2.0f);
						cv::Point2f b(eyes[1].x + eyes[1].width / 2.0f, eyes[1].y + eyes[1].height / 2.0f);
						if (a.x > b.x) std::swap(a, b);
						out.angle = std::atan2(b.y - a.y, b.x - a.x);
					}
				}
			}
		} catch (const cv::Exception &e) { obs_log(LOG_WARNING, "Detection error: %s", e.what()); }
		{ std::lock_guard<std::mutex> lock(f->result_mutex); f->target = out; }
		{ std::lock_guard<std::mutex> lock(f->work_mutex); f->worker_busy = false; }
	}
}

static void unload_image(FaceStickerFilter *f)
{
	if (!f->image_loaded) return;
	obs_enter_graphics(); gs_image_file_free(&f->image); obs_leave_graphics();
	f->image = {}; f->image_loaded = false;
}

static void load_image(FaceStickerFilter *f)
{
	unload_image(f);
	char *module_path = nullptr;
	std::string path;
	if (f->sticker == "custom") path = f->custom_path;
	else { module_path = obs_module_file(("stickers/" + f->sticker + ".png").c_str()); if (module_path) path = module_path; }
	if (!path.empty()) {
		gs_image_file_init(&f->image, path.c_str());
		obs_enter_graphics(); gs_image_file_init_texture(&f->image); obs_leave_graphics();
		f->image_loaded = f->image.loaded && f->image.texture;
		if (!f->image_loaded) obs_log(LOG_WARNING, "Unable to load sticker: %s", path.c_str());
	}
	bfree(module_path);
}

static const char *filter_name(void *) { return obs_module_text("FaceStickerFilter"); }
static void filter_update(void *data, obs_data_t *s)
{
	auto *f = (FaceStickerFilter *)data;
	std::string id = obs_data_get_string(s, "sticker"), path = obs_data_get_string(s, "custom_path");
	bool reload = f->sticker != id || f->custom_path != path;
	f->sticker = id; f->custom_path = path;
	f->scale = (float)obs_data_get_double(s, "scale");
	f->offset_x = (float)obs_data_get_double(s, "offset_x") / 100.0f;
	f->offset_y = (float)obs_data_get_double(s, "offset_y") / 100.0f;
	f->smoothing = (float)obs_data_get_double(s, "smoothing");
	f->detect_every = (int)obs_data_get_int(s, "detect_every");
	if (reload) load_image(f);
}

static void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	auto *f = new FaceStickerFilter; f->source = source;
	char *face = obs_module_file("models/haarcascade_frontalface_default.xml");
	char *eye = obs_module_file("models/haarcascade_eye_tree_eyeglasses.xml");
	if (!face || !f->face_cascade.load(face)) obs_log(LOG_ERROR, "Bundled face detector failed to load");
	if (!eye || !f->eye_cascade.load(eye)) obs_log(LOG_WARNING, "Eye detector failed to load; rotation disabled");
	bfree(face); bfree(eye);
	char *effect_path = obs_module_file("face-sticker.effect"), *error = nullptr;
	obs_enter_graphics(); f->effect = gs_effect_create_from_file(effect_path, &error); obs_leave_graphics();
	if (!f->effect) obs_log(LOG_ERROR, "Effect failed to load: %s", error ? error : "unknown");
	bfree(error); bfree(effect_path);
	filter_update(f, settings); if (!f->image_loaded) load_image(f);
	f->worker = std::thread(tracking_worker, f);
	return f;
}

static void filter_destroy(void *data)
{
	auto *f = (FaceStickerFilter *)data;
	{ std::lock_guard<std::mutex> lock(f->work_mutex); f->stopping = true; }
	f->work_cv.notify_one(); if (f->worker.joinable()) f->worker.join();
	unload_image(f);
	if (f->effect) { obs_enter_graphics(); gs_effect_destroy(f->effect); obs_leave_graphics(); }
	delete f;
}

static obs_source_frame *filter_video(void *data, obs_source_frame *frame)
{
	auto *f = (FaceStickerFilter *)data;
	f->frame_width = frame->width; f->frame_height = frame->height;
	if (f->face_cascade.empty() || ++f->frame_counter % f->detect_every) return frame;
	{
		std::lock_guard<std::mutex> lock(f->work_mutex);
		if (f->worker_busy || f->work_pending) return frame;
		cv::Mat gray; if (!frame_to_gray(frame, gray)) return frame;
		f->pending_gray = std::move(gray); f->work_pending = true;
	}
	f->work_cv.notify_one(); return frame;
}

static void filter_tick(void *data, float seconds)
{
	auto *f = (FaceStickerFilter *)data; Detection target;
	{ std::lock_guard<std::mutex> lock(f->result_mutex); target = f->target; }
	float k = std::clamp((1 - f->smoothing) * seconds * 60, .01f, 1.0f);
	auto sm = [k](float a, float b) { return a + (b - a) * k; };
	if (target.found) {
		if (!f->current.found) f->current = target;
		else { f->current.x = sm(f->current.x, target.x); f->current.y = sm(f->current.y, target.y);
			f->current.w = sm(f->current.w, target.w); f->current.h = sm(f->current.h, target.h);
			f->current.angle = sm(f->current.angle, target.angle); f->current.found = true;
			f->current.timestamp_ns = target.timestamp_ns; }
	} else if (os_gettime_ns() - f->current.timestamp_ns > 500000000ULL) f->current.found = false;
}

static void filter_render(void *data, gs_effect_t *)
{
	auto *f = (FaceStickerFilter *)data;
	if (!f->effect || !f->image_loaded || !f->current.found || !f->frame_width || !f->frame_height) {
		obs_source_skip_video_filter(f->source); return;
	}
	Placement p = placement_for(f->sticker); Detection d = f->current;
	float cx = d.x + (p.x + f->offset_x) * d.w, cy = d.y + (p.y + f->offset_y) * d.h;
	float nw = p.width * f->scale * d.w;
	float aspect = f->image.cx ? (float)f->image.cy / f->image.cx : 1;
	float nh = nw * aspect * f->frame_width / f->frame_height;
	vec4 rect; vec4_set(&rect, cx, cy, nw, nh);
	gs_effect_set_vec4(gs_effect_get_param_by_name(f->effect, "sticker_rect"), &rect);
	gs_effect_set_float(gs_effect_get_param_by_name(f->effect, "sticker_rotation"), d.angle);
	gs_effect_set_texture(gs_effect_get_param_by_name(f->effect, "sticker"), f->image.texture);
	if (obs_source_process_filter_begin(f->source, GS_RGBA, OBS_NO_DIRECT_RENDERING))
		obs_source_process_filter_end(f->source, f->effect, 0, 0);
}

static obs_properties_t *filter_properties(void *)
{
	auto *p = obs_properties_create();
	auto *list = obs_properties_add_list(p, "sticker", obs_module_text("Sticker"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	for (auto id : sticker_ids) obs_property_list_add_string(list, obs_module_text(id), id);
	obs_property_list_add_string(list, obs_module_text("CustomSticker"), "custom");
	obs_properties_add_path(p, "custom_path", obs_module_text("CustomPath"), OBS_PATH_FILE,
		"Images (*.png *.jpg *.jpeg);;All files (*.*)", nullptr);
	obs_properties_add_float_slider(p, "scale", obs_module_text("Scale"), .25, 3, .01);
	obs_properties_add_float_slider(p, "offset_x", obs_module_text("OffsetX"), -100, 100, 1);
	obs_properties_add_float_slider(p, "offset_y", obs_module_text("OffsetY"), -100, 100, 1);
	obs_properties_add_float_slider(p, "smoothing", obs_module_text("Smoothing"), 0, .98, .01);
	obs_properties_add_int_slider(p, "detect_every", obs_module_text("DetectionInterval"), 1, 10, 1);
	return p;
}
static void filter_defaults(obs_data_t *s)
{
	obs_data_set_default_string(s, "sticker", "glasses"); obs_data_set_default_double(s, "scale", 1);
	obs_data_set_default_double(s, "smoothing", .82); obs_data_set_default_int(s, "detect_every", 3);
}
static obs_source_info info = {.id = "obs_face_sticker_filter", .type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO, .get_name = filter_name, .create = filter_create,
	.destroy = filter_destroy, .get_defaults = filter_defaults, .get_properties = filter_properties,
	.update = filter_update, .video_tick = filter_tick, .video_render = filter_render,
	.filter_video = filter_video};
}

bool obs_module_load(void)
{
	obs_register_source(&info); obs_log(LOG_INFO, "OBS Face Sticker loaded (version %s)", PLUGIN_VERSION); return true;
}
void obs_module_unload(void) { obs_log(LOG_INFO, "OBS Face Sticker unloaded"); }
