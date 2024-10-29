ddsm_dicom="DDSM/csv/dicom_info.csv"
calc_train="DDSM/csv/calc_case_description_train_set.csv"
calc_test="DDSM/csv/calc_case_description_test_set.csv"
mass_train="DDSM/csv/mass_case_description_train_set.csv"
mass_test="DDSM/csv/mass_case_description_test_set.csv"
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
LAYERS=[4,6,8]
IN_CHANNELS=[160,384,2560]
OUT_CHANNELS=768
MAX_Y=3841
MAX_X=2893
NUM_BBOX=60
EPOCHS=10
BATCH_SIZE=1
NUM_WORKERS=4
DECAY=1e-4