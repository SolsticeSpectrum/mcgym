// temp bisect of pumpkin nbt chunk parsing, deleted before merge
use std::io::Read;

use pumpkin_nbt::Nbt;
use pumpkin_nbt::deserializer::NbtReadHelperJava;
use serde::Deserialize;

fn clean_chunk() -> Vec<u8> {
    let data = std::fs::read("../worlds/spiral/region/r.0.-1.mca").unwrap();
    // find a chunk with sections, idx 1 had full data
    let idx = 1usize;
    let loc = u32::from_be_bytes(data[idx * 4..idx * 4 + 4].try_into().unwrap());
    assert!(loc != 0);
    let at  = (loc >> 8) as usize * 4096;
    let len = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
    let mut nbt = Vec::new();
    flate2::read::ZlibDecoder::new(&data[at + 5..at + 4 + len]).read_to_end(&mut nbt).unwrap();

    let mut tree = Nbt::read(&mut NbtReadHelperJava::new(std::io::Cursor::new(&nbt[..]))).unwrap();
    tree.root_tag.child_tags.retain(|k, _| {
        matches!(&**k, "DataVersion" | "xPos" | "yPos" | "zPos" | "Status"
            | "sections" | "Heightmaps" | "block_ticks" | "fluid_ticks"
            | "block_entities" | "isLightOn")
    });
    tree.root_tag.child_tags.iter().for_each(|(k, _)| println!("kept {k}"));
    Nbt::new(String::new(), tree.root_tag).write_unnamed().to_vec()
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct JustVersion {
    data_version: i32,
}

#[derive(Deserialize, Debug)]
struct MiniSection {
    y: i8,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct WithSections {
    data_version: i32,
    #[serde(rename = "sections")]
    sections: Vec<MiniSection>,
}

#[derive(Deserialize, Debug)]
struct MiniBlockStates {
    palette: Vec<MiniPaletteEntry>,
    data:    Option<Box<[i64]>>,
}

#[derive(Deserialize, Debug)]
struct MiniPaletteEntry {
    #[serde(rename = "Name")]
    name: String,
}

#[derive(Deserialize, Debug)]
struct DeepSection {
    y:            i8,
    block_states: Option<MiniBlockStates>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct WithDeepSections {
    data_version: i32,
    #[serde(rename = "sections")]
    sections: Vec<DeepSection>,
}

#[test]
fn bisect() {
    let bytes = clean_chunk();

    let v: Result<JustVersion, _> =
        pumpkin_nbt::from_bytes_unnamed(std::io::Cursor::new(&bytes[..]));
    println!("just version: {:?}", v.is_ok());

    let v: Result<WithSections, _> =
        pumpkin_nbt::from_bytes_unnamed(std::io::Cursor::new(&bytes[..]));
    println!("mini sections: {:?}", v.as_ref().map(|c| c.sections.len()).map_err(|e| e.to_string()));

    let v: Result<WithDeepSections, _> =
        pumpkin_nbt::from_bytes_unnamed(std::io::Cursor::new(&bytes[..]));
    println!("deep sections: {:?}", v.as_ref().map(|c| c.sections.len()).map_err(|e| e.to_string()));
}
