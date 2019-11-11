import os
import bilsm_crf_model

project_path = os.path.dirname(__file__)
model_path = os.path.normpath(os.path.join(project_path, 'model/crf.h5'))

EPOCHS = 10
model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()
# train model
model.fit(train_x, train_y, batch_size=16, epochs=EPOCHS, validation_data=[test_x, test_y])
model.save(model_path)
