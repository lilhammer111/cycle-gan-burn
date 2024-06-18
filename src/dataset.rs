use std::fs::File;
use std::path::{Path, PathBuf};
use burn::data::dataset::vision::ImageFolderDataset;
use zip::ZipArchive;

pub trait Horse2ZebraLoader {
    fn horse2zebra_train_a() -> Self;
    fn horse2zebra_train_b() -> Self;
    fn horse2zebra_test_a() -> Self;
    fn horse2zebra_test_b() -> Self;
}

impl Horse2ZebraLoader for ImageFolderDataset {
    fn horse2zebra_train_a() -> Self {
        let root = Path::new(file!())
            .parent().unwrap()  // project src dir path
            .parent().unwrap();  // project root dir path

        let dpath = root.join("data/trainA");

        Self::new_classification(dpath).unwrap()
    }

    fn horse2zebra_train_b() -> Self {
        let root = Path::new(file!())
            .parent().unwrap()  // project src dir path
            .parent().unwrap();  // project root dir path

        let dpath = root.join("data/trainB");

        Self::new_classification(dpath).unwrap()
    }

    fn horse2zebra_test_a() -> Self {
        let root = Path::new(file!())
            .parent().unwrap()  // project src dir path
            .parent().unwrap();  // project root dir path

        let dpath = root.join("data/testA");

        Self::new_classification(dpath).unwrap()
    }

    fn horse2zebra_test_b() -> Self {
        let root = Path::new(file!())
            .parent().unwrap()  // project src dir path
            .parent().unwrap();  // project root dir path

        let dpath = root.join("data/testB");

        Self::new_classification(dpath).unwrap()
    }
}

/// Get the project root directory.
fn proj_root_dir() -> PathBuf {
    let root = Path::new(file!())
        .parent().unwrap()  // project src dir path
        .parent().unwrap();  // project root dir path

    root.to_path_buf()
}


/// Unzip the file to the specified path.
fn unzip(path: &str) -> PathBuf {
    let root = proj_root_dir();
    let file = File::open(root.join("archive.zip")).unwrap();

    let mut archive = ZipArchive::new(file).unwrap();

    archive.extract(root.join(&path)).unwrap();

    root.join(&path)
}

mod tests {
    use std::path::{Path};
    use burn::data::dataloader::Dataset;
    use burn::data::dataset::vision::ImageFolderDataset;
    use crate::dataset::{Horse2ZebraLoader, unzip, proj_root_dir};

    #[test]
    fn test_data_size() {
        assert_eq!(ImageFolderDataset::horse2zebra_train_a().len(), 1067);
        assert_eq!(ImageFolderDataset::horse2zebra_train_b().len(), 1334);
        assert_eq!(ImageFolderDataset::horse2zebra_test_a().len(), 120);
        assert_eq!(ImageFolderDataset::horse2zebra_test_b().len(), 140);
    }

    #[test]
    fn test_unzip() {
        assert_eq!(unzip("data"), Path::new("data"))
    }

    #[test]
    fn test_proj_root() {
        assert_eq!(proj_root_dir().as_path(), Path::new(""));
    }
}