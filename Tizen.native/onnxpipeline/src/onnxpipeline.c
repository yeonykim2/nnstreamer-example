#include "onnxpipeline.h"
#include <app.h>
#include <dlog.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <nnstreamer-single.h>
#include <nnstreamer.h>
#include <tizen_error.h>

typedef struct appdata {
	Evas_Object *win;
	Evas_Object *conform;
	Evas_Object *label;
	Evas_Object *table;
	Evas_Object *inference_button;
	Evas_Object *image;
	Evas_Object *box;
	Evas_Object *inference_exit;

} appdata_s;

const char *sample_image_filename = "cat.png";
const gchar *sample_image_path;
gchar *pipeline;
gchar *model;

ml_pipeline_src_h src_handle;
ml_pipeline_h pipeline_handle;
ml_pipeline_sink_h sink_handle;
ml_tensors_info_h info;
ml_tensors_data_h input;

static void
win_delete_request_cb(void *data, Evas_Object *obj, void *event_info)
{
	ui_app_exit();
}

static void
win_back_cb(void *data, Evas_Object *obj, void *event_info)
{
	appdata_s *ad = data;
	/* Let window go to hide state. */
	elm_win_lower(ad->win);
}

/**
 * @brief evas object smart callback
 */
static void
inference_button_cb (void *data, Evas_Object * obj, void *event_info)
{
	appdata_s *ad = data;
	int error_code;

	ml_tensor_dimension in_dim = {3, 720, 720, 1};

	ml_tensors_info_create(&info);
	ml_tensors_info_set_count(info, 1);
	ml_tensors_info_set_tensor_type(info, 0, ML_TENSOR_TYPE_UINT8);
	ml_tensors_info_set_tensor_dimension(info, 0, in_dim);

	error_code = ml_tensors_data_create(info, &input);
	if (error_code != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_tensors_data_create %d", error_code);
	}

	g_autofree gchar *bus_raw_file = g_strdup_printf ("%s",sample_image_path);
	if (!g_file_test (bus_raw_file, G_FILE_TEST_EXISTS)) {
		dlog_print(DLOG_ERROR, LOG_TAG, "do not load bus_raw_file");
	}

	void *bus_raw_data_buf = NULL;
	size_t bus_raw_data_size = 0;
	if (!g_file_get_contents (bus_raw_file, (gchar **) &bus_raw_data_buf, &bus_raw_data_size, NULL)) {
		dlog_print(DLOG_ERROR, LOG_TAG, "do not get g_file_get_contents");
	}

	error_code =
		ml_tensors_data_set_tensor_data(input, 0, bus_raw_data_buf, bus_raw_data_size);
	if (error_code != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_tensors_data_set_tensor_data %d",
				error_code);
	}

	error_code = ml_pipeline_src_input_data(src_handle, input,
											ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
	if (error_code != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_pipeline_src_input_data %d",
				error_code);
	}

	g_free (bus_raw_data_buf);
}

static void
create_base_gui(appdata_s *ad)
{
	/* Window */
	/* Create and initialize elm_win.
	   elm_win is mandatory to manipulate window. */
	ad->win = elm_win_util_standard_add(PACKAGE, PACKAGE);
	elm_win_autodel_set(ad->win, EINA_TRUE);

	if (elm_win_wm_rotation_supported_get(ad->win)) {
		int rots[4] = { 0, 90, 180, 270 };
		elm_win_wm_rotation_available_rotations_set(ad->win, (const int *)(&rots), 4);
	}

	evas_object_smart_callback_add(ad->win, "delete,request", win_delete_request_cb, NULL);
	eext_object_event_callback_add(ad->win, EEXT_CALLBACK_BACK, win_back_cb, ad);

	/* Conformant */
	/* Create and initialize elm_conformant.
	   elm_conformant is mandatory for base gui to have proper size
	   when indicator or virtual keypad is visible. */
	ad->conform = elm_conformant_add(ad->win);
	elm_win_indicator_mode_set(ad->win, ELM_WIN_INDICATOR_SHOW);
	elm_win_indicator_opacity_set(ad->win, ELM_WIN_INDICATOR_OPAQUE);
	evas_object_size_hint_weight_set(ad->conform, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
	elm_win_resize_object_add(ad->win, ad->conform);
	evas_object_show(ad->conform);

	ad->box = elm_box_add (ad->conform);
	evas_object_size_hint_weight_set (ad->box, EVAS_HINT_EXPAND,
		EVAS_HINT_EXPAND);
	elm_object_content_set (ad->conform, ad->box);

	ad->table = elm_table_add (ad->box);
	elm_table_homogeneous_set (ad->table, EINA_TRUE);
	evas_object_size_hint_weight_set (ad->table, EVAS_HINT_EXPAND,
		EVAS_HINT_EXPAND);
	evas_object_size_hint_align_set (ad->table, EVAS_HINT_FILL, EVAS_HINT_FILL);
	elm_box_pack_end (ad->box, ad->table);
	evas_object_show (ad->table);

	ad->image = elm_image_add (ad->table);
	evas_object_size_hint_align_set (ad->image, EVAS_HINT_FILL, EVAS_HINT_FILL);
	evas_object_size_hint_weight_set (ad->image, EVAS_HINT_EXPAND,
		EVAS_HINT_EXPAND);
	elm_image_file_set (ad->image, sample_image_path, NULL);
	elm_table_pack (ad->table, ad->image, 0, 0, 1, 1);
	evas_object_show (ad->image);

	ad->inference_button = elm_button_add (ad->box);
	evas_object_size_hint_align_set (ad->inference_button, EVAS_HINT_FILL,
		EVAS_HINT_FILL);
	elm_object_text_set (ad->inference_button, "Inference with trained model");
	evas_object_show (ad->inference_button);
	evas_object_smart_callback_add (ad->inference_button, "clicked",
		inference_button_cb, ad);
	elm_box_pack_end (ad->box, ad->inference_button);

	ad->inference_exit = elm_button_add (ad->box);
	evas_object_size_hint_align_set (ad->inference_exit, EVAS_HINT_FILL,
		EVAS_HINT_FILL);
	elm_object_text_set (ad->inference_exit, "Exit");
	evas_object_show (ad->inference_exit);
	evas_object_smart_callback_add (ad->inference_exit, "clicked",
		win_delete_request_cb, ad);
	elm_box_pack_end (ad->box, ad->inference_exit);

	/* Show window after base gui is set up */
	evas_object_show(ad->box);
	evas_object_show(ad->win);
}

/**
 * @brief New data callback function
 */
static void new_data_cb(const ml_tensors_data_h data,
                        const ml_tensors_info_h info, void *user_data) {
  char *data_ptr;
  size_t data_size;
  int error_code =
      ml_tensors_data_get_tensor_data(data, 0, (void **)&data_ptr, &data_size);

  if (error_code != ML_ERROR_NONE)
    dlog_print(DLOG_ERROR, LOG_TAG, "Failed to get tensor data.");

//   ecore_pipe_write(data_output_pipe, data_ptr, data_size);
}

static bool
app_create(void *data)
{
	/* Hook to take necessary actions before main event loop starts
		Initialize UI resources and application's data
		If this function returns true, the main loop of application starts
		If this function returns false, the application is terminated */
	appdata_s *ad = data;

	create_base_gui(ad);

	char *path = app_get_resource_path();
	model = g_build_filename(path, "candy.onnx", NULL);
	pipeline = g_strdup_printf(
		"appsrc name=appsrc ! "
		"other/tensor,type=uint8,dimension=1:720:720:3,framerate=0/1 ! "
		"tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! "
		"tensor_transform mode=arithmetic "
		"option=typecast:float32,add:0.0 ! "
		"tensor_filter framework=onnxruntime model=%s ! "
		"tensor_converter ! tensor_transform mode=transpose option=2:0:1:3 ! "
		"tensor_transform mode=typecast option=uint8 ! "
		"other/tensor,type=uint8,dimension=3:720:720:1,framerate=0/1 ! "
		"tensor_decoder mode=direct_video ! videoconvert ! "
		"video/x-raw,width=720,height=720,format=RGB,framerate=0/1 ! "
		"tensor_converter ! tensor_sink name=tensor_sink",
		model);

	// pipeline = g_strdup_printf(
	// "appsrc name=appsrc ! "
    //   "other/tensor,type=uint8,dimension=3:600:400:1,framerate=0/1 ! "
    //   "tensor_transform mode=arithmetic "
    //   "option=typecast:float32,add:0,div:255.0 ! "
    //   "tensor_filter name=tfilter framework=tensorflow-lite model=%s ! "
    //   "tensor_transform mode=arithmetic option=mul:255.0,add:0.0 ! "
    //   "tensor_transform mode=clamp option=0:255 ! "
    //   "other/tensor,type=float32,dimension=3:600:400:1,framerate=0/1 ! "
    //   "tensor_transform mode=typecast option=uint8 ! "
    //   "other/tensor,type=uint8,dimension=3:600:400:1,framerate=0/1 ! "
    //   "tensor_decoder mode=direct_video ! videoconvert ! "
    //   "video/x-raw,width=600,height=400,format=BGRA,framerate=0/1 ! "
    //   "tensor_converter ! tensor_sink name=tensor_sink",
    //   model);

	// pipeline = g_strdup_printf(
	// "v4l2src name=cam_src ! videoconvert ! videoscale ! video/x-raw,width=720,height=720,format=RGB ! "
	// "tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! "
	// "tensor_transform mode=arithmetic option=typecast:float32,add:0.0 ! "
	// "tensor_filter framework=onnxruntime model=%s ! "
	// "tensor_converter ! tensor_transform mode=transpose option=2:0:1:3 ! "
	// "tensor_transform mode=arithmetic option=typecast:uint8,add:0.0 ! "
	// "tensor_decoder mode=direct_video ! videoconvert ! "
	// "video/x-raw,width=720,height=720,format=RGB,framerate=0/1 ! "
	// "tensor_converter ! tensor_sink name=tensor_sink", model);

	// pipeline = g_strdup_printf(
	// "v4l2src name=cam_src ! videoconvert ! videoscale ! video/x-raw,width=720,height=720,format=RGB ! tee name=t_raw "
    //    "t_raw. ! queue ! videoconvert ! ximagesink name=img_origin sync=false"
    //    "t_raw. ! queue leaky=2 max-size-buffers=10 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! "
    //    "tensor_transform mode=arithmetic option=typecast:float32,add:0.0 ! "
	//    "tensor_filter framework=onnxruntime model=%s ! "
    //    "tensor_converter ! tensor_transform mode=transpose option=2:0:1:3 ! "
	//    "tensor_transform mode=arithmetic option=typecast:uint8,add:0.0 ! "
    //    "tensor_decoder mode=direct_video ! videoconvert ! ximagesink name=candy_img sync=false ", model);

	dlog_print(DLOG_ERROR, LOG_TAG, "model path= %s", path);
	int error_code =
		ml_pipeline_construct(pipeline, NULL, NULL, &pipeline_handle);
	if (error_code != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_pipeline_construct %d", error_code);
	}

	error_code =
		ml_pipeline_src_get_handle(pipeline_handle, "appsrc", &src_handle);
	if (error_code != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_pipeline_src_get_handle %d",
				error_code);
	}

	error_code = ml_pipeline_sink_register(pipeline_handle, "tensor_sink",
											new_data_cb, NULL, &sink_handle);
	if (error_code != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_pipeline_sink_register %d", error_code);
	}

	error_code = ml_pipeline_start(pipeline_handle);
	if (error_code != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_pipeline_start %d", error_code);
		ml_pipeline_destroy(pipeline_handle);
	}

	g_free(pipeline);
	g_free(model);

	return true;
}

static void
app_control(app_control_h app_control, void *data)
{
	/* Handle the launch request. */
}

static void
app_pause(void *data)
{
	/* Take necessary actions when application becomes invisible. */
}

static void
app_resume(void *data)
{
	/* Take necessary actions when application becomes visible. */
}

static void
app_terminate(void *data)
{
	/* Release all resources. */
}

static void
ui_app_lang_changed(app_event_info_h event_info, void *user_data)
{
	/*APP_EVENT_LANGUAGE_CHANGED*/

	int ret;
	char *language;

	ret = app_event_get_language(event_info, &language);
	if (ret != APP_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "app_event_get_language() failed. Err = %d.", ret);
		return;
	}

	if (language != NULL) {
		elm_language_set(language);
		free(language);
	}
}

static void
ui_app_orient_changed(app_event_info_h event_info, void *user_data)
{
	/*APP_EVENT_DEVICE_ORIENTATION_CHANGED*/
	return;
}

static void
ui_app_region_changed(app_event_info_h event_info, void *user_data)
{
	/*APP_EVENT_REGION_FORMAT_CHANGED*/
}

static void
ui_app_low_battery(app_event_info_h event_info, void *user_data)
{
	/*APP_EVENT_LOW_BATTERY*/
}

static void
ui_app_low_memory(app_event_info_h event_info, void *user_data)
{
	/*APP_EVENT_LOW_MEMORY*/
}

int
main(int argc, char *argv[])
{
	appdata_s ad = {0,};
	int ret = 0;
	g_autofree gchar *img_path = app_get_resource_path ();

	ui_app_lifecycle_callback_s event_callback = {0,};
	app_event_handler_h handlers[5] = {NULL, };

	sample_image_path =
		g_strdup_printf ("%s%s", img_path, sample_image_filename);
	dlog_print(DLOG_INFO, LOG_TAG, "img path is %s", sample_image_path);

	event_callback.create = app_create;
	event_callback.terminate = app_terminate;
	event_callback.pause = app_pause;
	event_callback.resume = app_resume;
	event_callback.app_control = app_control;

	ui_app_add_event_handler(&handlers[APP_EVENT_LOW_BATTERY], APP_EVENT_LOW_BATTERY, ui_app_low_battery, &ad);
	ui_app_add_event_handler(&handlers[APP_EVENT_LOW_MEMORY], APP_EVENT_LOW_MEMORY, ui_app_low_memory, &ad);
	ui_app_add_event_handler(&handlers[APP_EVENT_DEVICE_ORIENTATION_CHANGED], APP_EVENT_DEVICE_ORIENTATION_CHANGED, ui_app_orient_changed, &ad);
	ui_app_add_event_handler(&handlers[APP_EVENT_LANGUAGE_CHANGED], APP_EVENT_LANGUAGE_CHANGED, ui_app_lang_changed, &ad);
	ui_app_add_event_handler(&handlers[APP_EVENT_REGION_FORMAT_CHANGED], APP_EVENT_REGION_FORMAT_CHANGED, ui_app_region_changed, &ad);

	ret = ui_app_main(argc, argv, &event_callback, &ad);
	if (ret != APP_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "app_main() is failed. err = %d", ret);
	}

	return ret;
}
