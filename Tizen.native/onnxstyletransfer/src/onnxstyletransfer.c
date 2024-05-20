#include "onnxstyletransfer.h"
#include <glib.h>
#include <glib/gstdio.h>
#include <nnstreamer.h>
#include <ml-api-service.h>
#include <ml-api-common.h>

typedef struct appdata {
	Evas_Object *win;
	Evas_Object *conform;
	Evas_Object *box;
	Evas_Object *label;
	Evas_Object *table;
	Evas_Object *inference_button;
	Evas_Object *image;
	Evas_Object *inference_exit;

	g_autofree gchar *model_config;

	ml_tensors_info_h input_info;
	ml_tensors_info_h output_info;

} appdata_s;

const char *sample_image_filename = "cat.png";
const gchar *sample_image_path;
Ecore_Pipe *data_output_pipe;

// g_autofree gchar *img_path;

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

// ml_handle_h handle, const char *name, const ml_tensors_data_h new_data, void *user_data
/** @brief new_data callback for this app */
static void
_cb_new_data (ml_service_event_e event, ml_information_h event_data, void *user_data)
{
  appdata_s *app_data = (appdata_s *) user_data;
  ml_tensors_data_h data = NULL;
  void *_raw = NULL;
  size_t _size = 0;
  int status;

  switch (event) {
    case ML_SERVICE_EVENT_NEW_DATA:
      status = ml_information_get (event_data, "data", &data);
	if (status != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_information_get %d", status);
	}
      status = ml_tensors_data_get_tensor_data (data, 0U, &_raw, &_size);
	  if (status != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_tensors_data_get_tensor_data %d", status);
	}
	ecore_pipe_write(data_output_pipe, _raw, _size);
      break;
    default:
      break;
  }

}

/**
 * @brief evas object smart callback
 */
static void
inference_button_cb (void *data, Evas_Object * obj, void *event_info)
{
	appdata_s *ad = data;
	ml_service_h handle = NULL;
	int status;

	dlog_print(DLOG_INFO, LOG_TAG, "hi inference button");
  	// // set callbacks
	dlog_print(DLOG_INFO, LOG_TAG, "hi model config %s", ad->model_config);
	status = ml_service_new (ad->model_config, &handle);
	if (status != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_service_create %d", status);
	}
	// status = ml_service_destroy (handle);
	status = ml_service_set_event_cb (handle, _cb_new_data, &ad);
	if (status != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_service_set_callback %d", status);
	}

	// check input_info
	status = ml_service_get_input_information (handle, "camsrc", &ad->input_info);
	if (status != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_handle_get_input_information %d", status);
	}

	// check output_info
	status = ml_service_get_output_information (handle, "result_sink", &ad->output_info);
	if (status != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_service_get_input_information %d", status);
	}

	g_autofree gchar *bus_raw_file = g_strdup_printf ("%s",sample_image_path);
	if (!g_file_test (bus_raw_file, G_FILE_TEST_EXISTS)) {
		dlog_print(DLOG_ERROR, LOG_TAG, "do not load bus_raw_file");
	}

	uint8_t *bus_raw_data_buf = NULL;
	gsize bus_raw_data_size = 0;
	if (!g_file_get_contents (bus_raw_file, (gchar **) &bus_raw_data_buf, &bus_raw_data_size, NULL)) {
		dlog_print(DLOG_ERROR, LOG_TAG, "do not get g_file_get_contents");
	}

	// create tensors_data handle with input_info
	ml_tensors_data_h input_data = NULL;
	status = ml_tensors_data_create(ad->input_info, &input_data);
	if (status != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_tensors_data_create %d", status);
	}

	status =
		ml_tensors_data_set_tensor_data(input_data, 0, bus_raw_data_buf, bus_raw_data_size);
	if (status != ML_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "ml_tensors_data_set_tensor_data %d",
				status);
	}

	for (int i = 0; i < 5; i++) {
		g_usleep (50000U);

		status = ml_service_request (handle, "camsrc", input_data);
	}

	g_free (bus_raw_data_buf);
	ml_service_set_event_cb (handle, NULL, NULL);
	// destroy the tensors_data
	ml_tensors_data_destroy (input_data);

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

static bool
app_create(void *data)
{
	/* Hook to take necessary actions before main event loop starts
		Initialize UI resources and application's data
		If this function returns true, the main loop of application starts
		If this function returns false, the application is terminated */
	appdata_s *ad = data;

	create_base_gui(ad);

	int status;
	g_autofree gchar *res_path = app_get_resource_path ();

	ad->model_config = g_strdup_printf ("%s%s", res_path, "onnx_style_transfer.pipeline.appsrc.conf");
	dlog_print (DLOG_INFO, LOG_TAG, "model_config:%s", ad->model_config);

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
	ecore_pipe_del(data_output_pipe);
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

/**
 * @brief Update the image
 * @param data The data passed from the callback registration function (not used
 * here)
 * @param buf The data passed from pipe
 * @param size The data size
 */
void update_gui(void *data, void *buf, unsigned int size) {
  appdata_s *ad = data;
  unsigned char *img_src = evas_object_image_data_get(ad->image, EINA_TRUE);
  memcpy(img_src, buf, size);
  evas_object_image_data_update_add(ad->image, 0, 0, 720, 720);
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

	data_output_pipe = ecore_pipe_add(update_gui, NULL);

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
