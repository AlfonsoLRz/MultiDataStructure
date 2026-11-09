"""Readable names for every schema, library configuration, cell and device that reaches the paper.

The engine and the batteries name schemas compactly (kd48l128sm, qt9_ot24l32, qu4_hg12l2048,
e9pick_qt24l64, octree_default); the paper never prints those. label() turns any of them into words:

    kd48l128sm            kd-tree (sliding midpoint), leaf 128
    qt24l64tc             quadtree, leaf 64
    octree_default        octree, leaf 32 (library default)
    hg12l2048             hierarchical grid (4x4x4), leaf 2048
    str64l512             STR (64 children), leaf 512
    ot2_kd48l128sm        octree (2 levels) over kd-tree (sliding midpoint), leaf 128
    qu4_hg12l2048         quadtree (4 levels) over hierarchical grid (4x4x4), leaf 2048
    e9pick_qt24l64        quadtree, leaf 64   (with role="searched" -> "searched best single")

Depth is left out where it is open-ended (the last block of every searched schema), because it only
says that the tree may go as deep as the data needs; a prefix block keeps its level count. tex() wraps
the result for LaTeX. mapping_rows() lists label -> compact name for the supplementary table.
"""
import re

PRIMITIVES = {
    "kd": "kd-tree", "qt": "quadtree", "qu": "quadtree", "ot": "octree", "oc": "octree", "hg": "hierarchical grid (4x4x4)",
    "rg": "regular grid (3x3x3)", "bvh": "BVH", "bih": "bounding interval hierarchy", "lbvh": "LBVH", "kot": "Karras octree",
}
SPLIT_WORDS = {"sm": "sliding midpoint", "rr": "round robin", "cl": "cell centre", "is": "shortest axis ignored"}
DEFAULTS = {
    "octree_default": "octree, leaf 32 (library default)",
    "kdtree_default": "kd-tree (median split), leaf 32 (library default)",
    "quadtree_default": "quadtree, leaf 32 (library default)",
    "bvh_default": "BVH, leaf 32 (library default)",
    "bih_default": "bounding interval hierarchy, leaf 32 (library default)",
    "hgrid_default": "hierarchical grid, leaf 32 (library default)",
    "regular_grid_default": "regular grid, leaf 32 (library default)",
    "lbvh_default": "LBVH, leaf 32 (library default)",
    "karras_octree_default": "Karras octree, leaf 32 (library default)",
    "quadtree_octree": "quadtree (4 levels) over octree, leaf 32 (hand-written)",
    "octree_kdtree": "octree (4 levels) over kd-tree, leaf 32 (hand-written)",
}
LIBRARIES = {
    "nanoflann_kd_leaf": "nanoflann kd-tree, leaf {}",
    "unibn_octree_bucket": "unibn octree, bucket {}",
    "boost_rtree_rstar": "Boost R-tree, {} per node",
    "pcl_octree_res": "PCL octree, voxel = extent / {}",
}
LIBRARY_FIXED = {"pcl_kdtree_flann": "PCL kd-tree (FLANN)", "open3d_kdtreeflann": "Open3D kd-tree (FLANN)"}
SCENES = {"domfountain": "DomFountain", "galiciaforest": "GaliciaForest", "galiciaforest_es05": "GaliciaForest plot",
          "sanandreas": "SanAndreas", "solarplantation": "SolarPlantation", "waikato_hamilton": "Waikato",
          "ottobrunncampus": "OttobrunnCampus", "parislux": "ParisLux", "lille": "Lille", "alhambra": "Alhambra",
          "cathedral": "Cathedral", "neugasse": "Neugasse", "toronto": "Toronto", "aneto": "Aneto", "barcelona": "Barcelona",
          "retiro": "Retiro", "museum": "Museum"}
DEVICES = {"memory": "in memory", "nvme": "NVMe SSD", "sata": "SATA SSD", "qlc": "QLC SSD", "hdd": "hard disk"}

BLOCK = re.compile(r"^(?P<kind>kot|lbvh|bvh|bih|str|kd|qt|qu|ot|oc|hg|rg)(?P<num>\d+)(?:f)?(?:l(?P<leaf>\d+))?(?P<suffix>.*)$")


def _block_words(kind, num, suffix, last):
    if kind == "str":
        words = f"STR ({num} children)" if num else "STR"
        return words
    notes = []
    words = PRIMITIVES.get(kind, kind)
    if kind == "kd":
        words = "kd-tree"
        split = "sliding midpoint" if "sm" in suffix else ("round robin" if "rr" in suffix else ("cell centre" if "cl" in suffix else None))
        if split:
            notes.append(split)
        elif last:
            notes.append("median split")
    if not last and kind not in ("hg", "rg"):
        notes.append(f"{num} level{'s' if num != '1' else ''}")
    return f"{words} ({', '.join(notes)})" if notes else words


def label(name, role=None):
    """Readable name of a schema or library configuration (see the module docstring)."""
    if role == "searched":
        return "searched best single"
    raw = name.replace("\\", "/").split("/")[-1]
    raw = re.sub(r"\.json$", "", raw)
    for prefix in ("e9pick_", "tuned_", "generated_", "evolved_", "xover_", "repair_"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    raw = re.sub(r"^g\d+_i\d+_", "", raw)
    raw = raw.replace("ap=", "")
    if raw in DEFAULTS:
        return DEFAULTS[raw]
    if raw in LIBRARY_FIXED:
        return LIBRARY_FIXED[raw]
    for prefix, template in LIBRARIES.items():
        if raw.startswith(prefix):
            return template.format(re.sub(r"\D.*$", "", raw[len(prefix):]))
    if raw.startswith("B") and "_" in raw:  # out-of-core stage B prefix: B<kind><levels>_<single>
        raw = raw[1:]
    blocks = raw.split("_")
    parts, leaf = [], None
    for i, block in enumerate(blocks):
        m = BLOCK.match(block)
        if not m:
            return name  # not a schema name: leave it as it is
        last = i == len(blocks) - 1
        if m.group("leaf"):
            leaf = m.group("leaf")
        parts.append(_block_words(m.group("kind"), m.group("num"), m.group("suffix"), last))
    text = " over ".join(parts)
    return f"{text}, leaf {leaf}" if leaf else text


def short(name):
    """Primitive and leaf only, for figure legends: 'kd-tree, leaf 128', 'octree over kd-tree, leaf 128'."""
    full = label(name)
    full = re.sub(r" \((?:sliding midpoint|median split|round robin|cell centre)?(?:, )?(?:\d+ levels?)?\)", "", full)
    full = re.sub(r" \((?:4x4x4|3x3x3)\)", "", full)
    return full.replace(" (library default)", " (default)")


def cell(name):
    """'domfountain_100M' -> 'DomFountain, 100M'."""
    m = re.match(r"^(?P<scene>.+)_(?P<size>\d+[MB])$", name)
    if not m:
        return name
    return f"{SCENES.get(m.group('scene'), m.group('scene'))}, {m.group('size')}"


def scene(name):
    m = re.match(r"^(?P<scene>.+)_(?P<size>\d+[MB])$", name)
    key = m.group("scene") if m else name
    return SCENES.get(key, key)


def device(name):
    return DEVICES.get(name, name)


def tex(text):
    return re.sub(r"([&%$#_{}])", r"\\\1", text)


def mapping_rows(names):
    """(readable label, compact name) pairs for the supplementary table, sorted by label."""
    return sorted({(label(n), n) for n in names})


if __name__ == "__main__":
    for n in ["kd48l128sm", "qt24l64tc", "octree_default", "kdtree_default", "hg12l2048", "str64l512", "ot2_kd48l128sm",
              "qu4_hg12l2048", "e9pick_qt24l64", "generated_qt6l128_kd48l128ap=sm", "qt9_ot24l32", "kd6_qt24l128tc",
              "bvh6l64_kd4l64ap=sm_qt24l64", "nanoflann_kd_leaf10", "boost_rtree_rstar16_str", "unibn_octree_bucket32",
              "pcl_kdtree_flann", "Boc2_kd48l512sm", "tuned_kd20l32", "qt16l128tc"]:
        print(f"{n:36} {label(n):60} | {short(n)}")
    print(cell("waikato_hamilton_100M"), "|", cell("galiciaforest_es05_100M"))
