# Release demonstration media

## JIT Editor

`jit-editor.gif` is the complete approximately 30-second recording, resized to
1280 pixels wide at 8 fps with an optimized shared palette and continuous looping.
It is approximately 1.8 MB and contains no audio. The original MP4 was not changed.

[The separate JIT release notes](../releases/JIT-Editor.md) embed it inline using
an absolute GitHub URL, which works after the file is committed and pushed to `main`.

## File Loader

The release notes use the short GIF inline, linked to the longer MP4 with audio.
Both are web copies of the supplied recordings; their originals were not changed.

- `auto-segmentation.gif`: the complete approximately 30-second animation,
  resized to 1080 pixels wide at 8 fps with an optimized shared palette.
- `auto-segmentation.mp4`: the complete approximately 3:06 recording, resized to
  1920 pixels wide at 12 fps. H.264 video, original AAC audio copied unchanged,
  and a fast-start container for playback while downloading.

The absolute GitHub image/video URLs in [the catalog release notes](../releases/Catalog.md)
work after these files have been committed and pushed to `main`. Local files
alone are not public uploads. The linked MP4 is a file download/view target;
it does not by itself create GitHub's inline attachment player.

### Optional inline MP4 player

Drag the MP4 into GitHub's Markdown editor for the release description or README.
GitHub uploads it and inserts an attachment URL. Keep that generated URL on its
own line below the GIF to include the video player, and copy it into the saved
release notes if future releases should reuse it. Uploading it as a release
download asset is a different operation from attaching it inside Markdown.

GitHub supports MP4, MOV and WebM video attachments and recommends H.264 for
browser compatibility. Attachment limits are 10 MB for GIFs/images, and 10 MB
for videos on free plans or 100 MB on paid plans. See
[GitHub's attachment documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files)
and [video support in repository Markdown](https://github.blog/changelog/2021-05-13-video-uploads-now-generally-available/).
