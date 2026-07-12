from pathlib import Path

from src.eval.evaluation_provenance import (
    build_manifest,
    fingerprint_tree,
    main,
    manifest_matches,
    write_manifest,
)


def test_manifest_detects_value_file_and_tree_changes(tmp_path: Path):
    source_file = tmp_path / "input.json"
    source_tree = tmp_path / "model"
    source_file.write_text("one")
    source_tree.mkdir()
    (source_tree / "weight.bin").write_bytes(b"weights")
    manifest_path = tmp_path / "provenance.json"

    expected = build_manifest(
        stage="generation",
        values={"state": "complete", "seed": "0"},
        files={"eval_json": str(source_file)},
        trees={"model": str(source_tree)},
    )
    write_manifest(manifest_path, expected)
    assert manifest_matches(manifest_path, expected) == (True, "")

    source_file.write_text("two")
    changed_file = build_manifest(
        stage="generation",
        values={"state": "complete", "seed": "0"},
        files={"eval_json": str(source_file)},
        trees={"model": str(source_tree)},
    )
    assert not manifest_matches(manifest_path, changed_file)[0]

    source_file.write_text("one")
    (source_tree / "weight.bin").write_bytes(b"different")
    changed_tree = build_manifest(
        stage="generation",
        values={"state": "complete", "seed": "0"},
        files={"eval_json": str(source_file)},
        trees={"model": str(source_tree)},
    )
    assert not manifest_matches(manifest_path, changed_tree)[0]


def test_tree_fingerprint_is_stable_and_tracks_symlinks(tmp_path: Path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "target-a").write_text("same")
    (tree / "target-b").write_text("same")
    (tree / "link").symlink_to("target-a")

    first = fingerprint_tree(tree)
    assert fingerprint_tree(tree) == first

    (tree / "link").unlink()
    (tree / "link").symlink_to("target-b")
    assert fingerprint_tree(tree) != first


def test_cli_check_and_atomic_write(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("content")
    manifest = tmp_path / "state/provenance.json"
    common = [
        "--manifest",
        str(manifest),
        "--stage",
        "preparation",
        "--value",
        "state=complete",
        "--file",
        f"source={source}",
    ]

    assert main(["write", *common]) == 0
    assert main(["check", *common]) == 0
    assert not list(manifest.parent.glob(f".{manifest.name}.tmp-*"))

    source.write_text("changed")
    assert main(["check", *common, "--quiet"]) == 1


def test_promote_rechecks_inputs_and_fingerprints_outputs(tmp_path: Path):
    source = tmp_path / "source.txt"
    output_dir = tmp_path / "videos"
    output = output_dir / "sample.mp4"
    source.write_text("input")
    output_dir.mkdir()
    output.write_bytes(b"video-one")
    manifest = tmp_path / "provenance.json"
    common = [
        "--manifest",
        str(manifest),
        "--stage",
        "generation",
        "--file",
        f"source={source}",
    ]

    assert main(["write", *common, "--value", "state=in_progress_resume"]) == 0
    source.write_text("changed")
    assert (
        main(
            [
                "promote",
                *common,
                "--value",
                "state=in_progress_resume",
                "--media-tree",
                f"videos={output_dir}",
                "--quiet",
            ]
        )
        == 1
    )

    source.write_text("input")
    assert (
        main(
            [
                "promote",
                *common,
                "--value",
                "state=in_progress_resume",
                "--media-tree",
                f"videos={output_dir}",
            ]
        )
        == 0
    )
    complete = [
        "check",
        *common,
        "--value",
        "state=complete",
        "--media-tree",
        f"videos={output_dir}",
    ]
    assert main(complete) == 0
    output.write_bytes(b"video-two")
    assert main([*complete, "--quiet"]) == 1
