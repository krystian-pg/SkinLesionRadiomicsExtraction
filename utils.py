import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
from radiomics import featureextractor
import logging
from augmentations import get_augmentations
from imblearn.over_sampling import SMOTE

# Set the pyradiomics logging level to ERROR to ignore warnings
radiomics_logger = logging.getLogger('radiomics')
radiomics_logger.setLevel(logging.ERROR)

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

def load_metadata(metadata_file):
    df = pd.read_csv(metadata_file)
    df['mask_id'] = df['image_id'].astype(str) + '_segmentation'
    df['cell_type'] = df['dx'].map(lesion_type_dict.get)
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
    return df

def map_paths(df, image_dirs, mask_dir):
    all_image_path = []
    for image_dir in image_dirs:
        all_image_path.extend(glob(os.path.join(image_dir, '*.jpg')))
    all_masks_path = glob(os.path.join(mask_dir, '*.png'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    maskid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_masks_path}
    df['image_path'] = df['image_id'].map(imageid_path_dict.get)
    df['mask_path'] = df['mask_id'].map(maskid_path_dict.get)
    return df

def split_data(df):
    df_undup = df.groupby('lesion_id').count()
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)
    unique_list = list(df_undup['lesion_id'])
    df['duplicates'] = df['lesion_id'].apply(lambda x: 'unduplicated' if x in unique_list else 'duplicated')
    df_undup = df[df['duplicates'] == 'unduplicated']
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=42, stratify=y)
    df_train = df[~df['image_id'].isin(df_val['image_id'])]
    return df_train, df_val

def augment_image(image_path, mask_path, transform):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, 0)
    transformed = transform(image=image, mask=mask)
    return transformed['image'], transformed['mask']

def extract_features(image, mask, extractor, mode='grayscale'):
    if mode == 'grayscale':
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sitk_image = sitk.GetImageFromArray(gray_image)
        mask = mask.astype(np.uint8)
        sitk_mask = sitk.GetImageFromArray(mask)
        features = extractor.execute(sitk_image, sitk_mask)
        return features
    elif mode == 'rgb':
        # Extract features from each RGB layer
        features = {}
        for color, suffix in zip(cv2.split(image), ['_r', '_g', '_b']):
            sitk_image = sitk.GetImageFromArray(color)
            sitk_mask = sitk.GetImageFromArray(mask)
            color_features = extractor.execute(sitk_image, sitk_mask)
            for key, value in color_features.items():
                features[key + suffix] = value
        return features
    else:
        raise ValueError("Invalid mode. Choose either 'grayscale' or 'rgb'.")

def generate_features(df, extractor, transform=None, mode='grayscale'):
    results = []
    for idx in tqdm(range(len(df)), desc="Processing"):
        image_path = df.iloc[idx]['image_path']
        mask_path = df.iloc[idx]['mask_path']
        if transform:
            image, mask = augment_image(image_path, mask_path, transform)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"Mask not found for {mask_path}, skipping...")
            continue
        features = extract_features(image, mask, extractor, mode)
        results.append(features)
    return results

def results_to_dataframe(results, df):
    # Filter out features that do not start with 'original_'
    feature_names = [key for key in results[0].keys() if key.startswith("original_")]
    samples = np.zeros((len(results), len(feature_names)))
    for i, result in enumerate(results):
        for j, feature in enumerate(feature_names):
            samples[i, j] = result[feature]
    df_features = pd.DataFrame(samples, columns=feature_names)
    df_combined = pd.concat([df.reset_index(drop=True), df_features], axis=1)
    return df_combined

def save_features(df_features, output_path):
    df_features.to_csv(output_path, index=False)

def process_test_set(image_dir, mask_dir, metadata_file, extractor, output_path, transform, mode='grayscale'):
    df_test = pd.read_csv(metadata_file)
    df_test['image_path'] = df_test['image_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
    df_test['mask_path'] = df_test['image_id'].apply(lambda x: os.path.join(mask_dir, f"{x}_segmentation.png"))

    results = []
    for idx in tqdm(range(len(df_test)), desc="Processing Test Set"):
        image_path = df_test.iloc[idx]['image_path']
        mask_path = df_test.iloc[idx]['mask_path']
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found or unable to read {image_path}, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"Mask not found for {mask_path}, skipping...")
            continue
        mask = mask.astype(np.uint8)  # Ensure mask is in a proper numerical type

        # Apply resize augmentation
        augmented = transform(image=image, mask=mask)
        image_resized = augmented['image']
        mask_resized = augmented['mask']

        try:
            features = extract_features(image_resized, mask_resized, extractor, mode=mode)
            results.append(features)
        except ValueError as e:
            print(f"No labels found in this mask {mask_path}, skipping...")

    df_test_features = results_to_dataframe(results, df_test.iloc[:len(results)])
    save_features(df_test_features, output_path)
