# Local model required

No model weight is distributed with this public staging repository. Supply a compatible LR-ASPP TorchScript file locally and pass its absolute path to the launcher with `model_path:=...`.

The expected model consumes an `N x 3 x H x W` tensor and returns an `N x C x H x W` semantic-logit tensor with white-boundary class `1` and yellow-centerline class `2`. Store the file outside this repository or under an ignored local path.
