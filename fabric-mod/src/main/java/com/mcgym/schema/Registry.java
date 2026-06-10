package com.mcgym.schema;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

// the mcgym id registry, maps resource names to the gyms integer ids, mapping
// by name keeps ids aligned regardless of client ordering
public final class Registry {

    private static final String[] SUFFIXES = {"_log", "_wood", "_stem", "_hyphae"};
    private static final String   EXCLUDED = "minecraft:mushroom_stem";

    private final Map<String, Integer> blocks;
    private final Map<String, Integer> items;
    private final Set<Integer>         logBlocks;
    private final Set<Integer>         logItems;

    private Registry(Map<String, Integer> blocks, Map<String, Integer> items) {
        this.blocks = blocks;
        this.items = items;
        this.logBlocks = logs(blocks);
        this.logItems = logs(items);
    }

    public static Registry load(Path json) throws IOException {
        JsonObject root = JsonParser.parseString(Files.readString(json, StandardCharsets.UTF_8)).getAsJsonObject();
        return new Registry(parse(root.getAsJsonObject("blocks")), parse(root.getAsJsonObject("items")));
    }

    private static Map<String, Integer> parse(JsonObject obj) {
        Map<String, Integer> out = new HashMap<>(obj.size() * 2);
        for (Map.Entry<String, JsonElement> e : obj.entrySet())
            out.put(e.getKey(), e.getValue().getAsInt());

        return out;
    }

    private static Set<Integer> logs(Map<String, Integer> ids) {
        Set<Integer> out = new HashSet<>();
        for (Map.Entry<String, Integer> e : ids.entrySet()) {
            String name = e.getKey();
            if (!name.startsWith("minecraft:") || name.equals(EXCLUDED)) continue;
            
            for (String s : SUFFIXES) {
                if (name.endsWith(s)) {
                    out.add(e.getValue());
                    break;
                }
            }
        }

        return out;
    }

    public int blockId(String name) {
        Integer id = blocks.get(name);
        return id != null ? id : 0;
    }

    public int itemId(String name) {
        Integer id = items.get(name);
        return id != null ? id : 0;
    }

    public boolean isLogBlock(int id) {
        return logBlocks.contains(id);
    }

    public boolean isLogItem(int id) {
        return logItems.contains(id);
    }

    public int blocks() {
        return blocks.size();
    }

    public int items() {
        return items.size();
    }
}
