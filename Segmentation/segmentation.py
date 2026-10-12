def label_training_dataset(training_path, region_path):
    files = os.listdir(training_path)


    for file in files:
        full_path = os.path.join(training_path, file)
        save_path = os.path.join(region_path, file + "_pos_mask.npy")

        if os.path.exists(save_path):
            continue

        img = mpimg.imread(full_path)
        fig = plt.figure()
        plt.imshow(img, interpolation='nearest', cmap="Greys")
        plt.colorbar()
        plt.title("First ROI")
        plt.show(block=False)



        if os.path.exists(save_path):
            continue


        # Let user draw first ROI
        roi1 = RoiPoly(color='b', fig=fig)

        # Show the image with the first ROI
        fig = plt.figure()
        plt.imshow(img, interpolation='nearest', cmap="Greys")
        plt.colorbar()
        roi1.display_roi()


        mask = roi1.get_mask(img[:, :, 0])
        np.save(save_path, mask)


        save_path = os.path.join(region_path, file + "_neg_mask.npy")
        plt.title('Second ROI (-ve)')
        plt.show(block=False)

        roi2 = RoiPoly(color='b', fig=fig)


        plt.imshow(img, interpolation='nearest', cmap="Greys")
        plt.colorbar()
        plt.title('The two ROIs')
        roi1.display_roi()
        roi2.display_roi()
        plt.show()

        mask = roi2.get_mask(img[:, :, 0])

        np.save(save_path, mask)




#hand label pos region for testing data
#write regions to files, this function should not return anything
def label_testing_dataset(training_path, region_path):
    files = os.listdir(training_path)

    for file in files:
        full_path = os.path.join(training_path, file)
        save_path = os.path.join(region_path, file + "_mask.npy")

        img = mpimg.imread(full_path)
        fig = plt.figure()
        plt.imshow(img, interpolation='nearest', cmap="Greys")
        plt.colorbar()
        plt.title("left click: line segment         right click or double click: close region")
        plt.show(block=False)


        if os.path.exists(save_path):
            continue

        # Let user draw first ROI
        roi1 = RoiPoly(color='b', fig=fig)

        # Show the image with the first ROI
        fig = plt.figure()
        plt.imshow(img, interpolation='nearest', cmap="Greys")
        plt.colorbar()
        roi1.display_roi()
        plt.title('ROI')
        plt.show()

        mask = roi1.get_mask(img[:, :, 0])

        np.save(save_path, mask)

def import_pre_labeled_training(training_path, region_path):
    features, labels = None, None

    files = os.listdir(training_path)

    all_barrel_pixels = []
    all_bg_pixels = []

    for file in files:
        full_path = os.path.join(training_path, file)
        pos_save_path = os.path.join(region_path, file + "_pos_mask.npy")
        neg_save_path = os.path.join(region_path, file + "_neg_mask.npy")
        masks = []

        pos_mask = np.load(pos_save_path)
        neg_mask = np.load(neg_save_path)


        img = mpimg.imread(full_path)

        barrel_pixels = img[pos_mask == 1]
        bg_pixels = img[neg_mask == 1]

        all_barrel_pixels.append(barrel_pixels)
        all_bg_pixels.append(bg_pixels)

    # stacking the list of barrel np arrays, and bg np arrays on top of each other
    positive_pixels = np.vstack(all_barrel_pixels)
    negative_pixels = np.vstack(all_bg_pixels)

    # dtype specified to not get floats
    # label = 1 per pos pixel, 0 per neg pixel
    positive_labels = np.ones(len(positive_pixels), dtype=int)
    negative_labels = np.zeros(len(negative_pixels), dtype=int)

    # stacking the positive pixels and negative pixels, labels horizontally concatenated in same order as features

    features = np.vstack([positive_pixels, negative_pixels])
    labels = np.concatenate([positive_labels, negative_labels])

    return features, labels

#import per hand labeled region for testing data
def import_pre_labeled_testing(testing_path, region_path):
    features, labels = None, None

    files = os.listdir(testing_path)

    all_barrel_pixels = []
    all_bg_pixels = []

    for file in files:
        full_path = os.path.join(testing_path, file)
        save_path = os.path.join(region_path, file + "_mask.npy")

        img = mpimg.imread(full_path)

        # handling jpg
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0


        mask = np.load(save_path)

        barrel_pixels = img[mask]
        bg_pixels = img[~mask]

        all_barrel_pixels.append(barrel_pixels)
        all_bg_pixels.append(bg_pixels)

    # stacking the list of barrel np arrays, and bg np arrays on top of each other
    positive_pixels = np.vstack(all_barrel_pixels)
    negative_pixels = np.vstack(all_bg_pixels)

    # dtype specified to not get floats
    # label = 1 per pos pixel, 0 per neg pixel
    positive_labels = np.ones(len(positive_pixels), dtype=int)
    negative_labels = np.zeros(len(negative_pixels), dtype=int)

    # stacking the positive pixels and negative pixels, labels horizontally concatenated in same order as features
    features = np.vstack([positive_pixels, negative_pixels])
    labels = np.concatenate([positive_labels, negative_labels])

    return features, labels



def train_GDA_common_variance(features, labels):
    prior, mu, cov = None, None, None

    positive_features = features[labels == 1]
    negative_features = features[labels == 0]

    mu_1 = np.mean(positive_features, axis=0)
    mu_2 = np.mean(negative_features, axis=0)

    mu = [mu_1, mu_2]

    total_positive = len(positive_features)
    total_negative = len(negative_features)
    total = total_positive + total_negative

    prior = [total_positive / total, total_negative / total]

    cov_1 = np.cov(positive_features, rowvar=False)
    cov_2 = np.cov(negative_features, rowvar=False)


    comb_cov = (total_positive * cov_1 + total_negative * cov_2) / total
    cov = [comb_cov]

    return prior, mu, cov

def train_GDA_variable_variance(features, labels):
    prior, mu, cov = None, None, None

    positive_features = features[labels == 1]
    negative_features = features[labels == 0]

    mu_1 = np.mean(positive_features, axis=0)
    mu_2 = np.mean(negative_features, axis=0)

    mu = [mu_1, mu_2]

    #print("MU", mu)

    total_positive = len(positive_features)
    total_negative = len(negative_features)
    total = total_positive + total_negative

    prior = [total_positive / total, total_negative / total]

    #print("prior", prior)

    cov_1 = np.cov(positive_features, rowvar=False)
    cov_2 = np.cov(negative_features, rowvar=False)
    cov = [cov_1, cov_2]

    #print(cov_1, cov_2)

    return prior, mu, cov

# def gaussian(x, mu, prior, cov, cov_inv, cov_det=None):
#
#     if cov_det is None:
#         return (-0.5 * ((x-mu).T @ cov_inv @ (x-mu))) + np.log(prior)

def gda_pred(x, mu_1, mu_0, log_prior_1, log_prior_0, cov_inv, cov_inv_0=None, other_term_1=None, other_term_2=None):
    prob_1 = (-0.5 * ((x - mu_1).T @ cov_inv @ (x - mu_1))) + log_prior_1

    if cov_inv_0 is None:
        prob_0 = (-0.5 * ((x-mu_0).T @ cov_inv @ (x-mu_0))) + log_prior_0
        return 1 if prob_1 > prob_0 else 0
    else:
        prob_1 = other_term_1 + prob_1
        prob_0 = other_term_2 + (-0.5 * ((x-mu_0).T @ cov_inv_0 @ (x-mu_0))) + log_prior_0
        return 1 if prob_1 > prob_0 else 0


#assign labels using trained GDA parameters for testing features
def predict(testing_features, theta, mu, cov):
    predicted_labels = None

    prior_1 = theta[0]
    prior_0 = theta[1]

    mu_1 = mu[0]
    mu_0 = mu[1]

    # all matrices are shifted about the means
    # diff_1 = testing_features - mu_1
    # diff_0 = testing_features - mu_0

    new_cov = cov[0]
    cov_inv = np.linalg.inv(new_cov)

    log_prior_1 = np.log(prior_1)
    log_prior_0 = np.log(prior_0)

    if len(cov) > 1:
        add_cov = cov[1]

        cov_det_1 = np.linalg.det(new_cov)
        cov_det_0 = np.linalg.det(add_cov)
        cov_inv_0 = np.linalg.inv(add_cov)

        other_factor_1 = -1.5 * np.log(2 * np.pi) + -0.5 * np.log(cov_det_1)
        other_factor_0 = -1.5 * np.log(2 * np.pi) + -0.5 * np.log(cov_det_0)


        predicted_labels = np.apply_along_axis(gda_pred, 1, testing_features, mu_1, mu_0, log_prior_1, log_prior_0, cov_inv, cov_inv_0, other_factor_1, other_factor_0).astype(int)

    else:
        predicted_labels = np.apply_along_axis(gda_pred, 1, testing_features, mu_1, mu_0, log_prior_1, log_prior_0, cov_inv).astype(int)

        # np.array([
        #     1 if (gaussian(x, mu_1, prior_1, new_cov, cov_inv)) > (gaussian(x, mu_0, prior_0, new_cov, cov_inv)) else 0
        #     for x in testing_features
        # ])

    # classifiying pixels to labels

    return predicted_labels

def segment_images(testing_path, prior, mu, cov, common):
    files = os.listdir(testing_path)

    for file in files:
        path = os.path.join(testing_path, file)

        img = mpimg.imread(path)

        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        # automatic dimension with -1
        pixels = img.reshape(-1, 3)


        labels = predict(pixels, prior, mu, cov)

        h, w, _ = img.shape
        segmented = np.zeros((h * w, 3), dtype=np.uint8)
        segmented[labels == 1] = [255, 0, 0]
        segmented = segmented.reshape(h, w, 3)

        name, ext = os.path.splitext(file)
        suffix = f"{name}_segmented{ext}"

        if common:
            out_path = os.path.join("segmentation_GDA_common", suffix)
            mpimg.imsave(out_path, segmented.astype(np.uint8))
        else:
            out_path = os.path.join("segmentation_GDA_variance", suffix)
            mpimg.imsave(out_path, segmented.astype(np.uint8))


def segment_training(training_path, prior, mu, cov, common, file):
    path = os.path.join(training_path, file)

    img = mpimg.imread(path)

    # automatic dimension with -1
    pixels = img.reshape(-1, 3)

    labels = predict(pixels, prior, mu, cov)

    h, w, _ = img.shape
    segmented = np.zeros((h * w, 3), dtype=np.uint8)
    segmented[labels == 1] = [255, 0, 0]
    segmented = segmented.reshape(h, w, 3)

    name, ext = os.path.splitext(file)

    if common:
        suffix = f"{name}_segmented_common{ext}"
    else:
        suffix = f"{name}_segmented_variance{ext}"


    out_path = os.path.join("training_segmentation", suffix)
    mpimg.imsave(out_path, segmented.astype(np.uint8))




def accuracy_analysis(predicted_labels, ground_truth_labels):
    # bitwise and returns only when both labels satisfy the conditions
    TP = np.sum((predicted_labels == 1) & (ground_truth_labels == 1))
    FP = np.sum((predicted_labels == 1) & (ground_truth_labels == 0))
    FN = np.sum((predicted_labels == 0) & (ground_truth_labels == 1))
    TN = np.sum((predicted_labels == 0) & (ground_truth_labels == 0))

    precision_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)

    precision_0 = TN / (TN + FN)
    recall_0 = TN / (TN + FP)


    print(f"precision of label 0: {precision_0:.2%}")
    print(f"recall of label 0: {recall_0:.2%}")
    print(f"precision of label 1: {precision_1:.2%}")
    print(f"recall of label 1: {recall_1:.2%}")


if __name__ == '__main__':
    training_features, training_labels = import_pre_labeled_training('trainset', 'train_region')
    testing_features, ground_truth_labels = import_pre_labeled_testing('testset', 'test_region')

    prior, mu, cov = train_GDA_common_variance(training_features, training_labels)

    segment_images("testset", prior, mu, cov, True)
    segment_training("trainset", prior, mu, cov, True, "7.3.png")

    predicted_labels = predict(testing_features, prior, mu, cov)

    print("GDA with common variance:")
    accuracy_analysis(predicted_labels, ground_truth_labels)

    prior, mu, cov = train_GDA_variable_variance(training_features, training_labels)

    segment_images("testset", prior, mu, cov, False)
    segment_training("trainset", prior, mu, cov, False, "7.3.png")

    predicted_labels = predict(testing_features, prior, mu, cov)

    print("GDA with variable variance:")
    accuracy_analysis(predicted_labels, ground_truth_labels)

