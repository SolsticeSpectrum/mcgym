package com.mcai.agent;

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

/**
 * Loads the MCAI id registry (schema/registry.json) and maps resource-location
 * names (e.g. "minecraft:oak_log") to the gym's integer ids. Mapping by NAME
 * keeps the ids aligned with the gym regardless of the client's internal
 * registry ordering.
 */
public final class SchemaRegistry {
    private final Map<String, Integer> blocks;
    private final Map<String, Integer> items;
    // Item ids whose resource name path contains "log" (oak_log, spruce_log, ...),
    // used for the wood-count telemetry. Precomputed at load.
    private final Set<Integer> logItemIds;

    private SchemaRegistry(Map<String, Integer> blocks, Map<String, Integer> items, Set<Integer> logItemIds) {
        this.blocks = blocks;
        this.items = items;
        this.logItemIds = logItemIds;
    }

    public static SchemaRegistry load(Path registryJson) throws IOException {
        String text = Files.readString(registryJson, StandardCharsets.UTF_8);
        JsonObject root = JsonParser.parseString(text).getAsJsonObject();
        Map<String, Integer> blocks = parse(root.getAsJsonObject("blocks"));
        Map<String, Integer> items = parse(root.getAsJsonObject("items"));
        Set<Integer> logItemIds = new HashSet<>();
        for (Map.Entry<String, Integer> e : items.entrySet()) {
            if (e.getKey().contains("log")) {
                logItemIds.add(e.getValue());
            }
        }
        return new SchemaRegistry(blocks, items, logItemIds);
    }

    private static Map<String, Integer> parse(JsonObject obj) {
        Map<String, Integer> out = new HashMap<>(obj.size() * 2);
        for (Map.Entry<String, com.google.gson.JsonElement> e : obj.entrySet()) {
            out.put(e.getKey(), e.getValue().getAsInt());
        }
        return out;
    }

    /** Block id for a resource name; 0 (air) when unknown. */
    public int blockId(String resourceName) {
        Integer id = blocks.get(resourceName);
        return id != null ? id : 0;
    }

    /** Item id for a resource name; 0 (empty) when unknown. */
    public int itemId(String resourceName) {
        Integer id = items.get(resourceName);
        return id != null ? id : 0;
    }

    /** Whether an item id corresponds to a log item (wood-count telemetry). */
    public boolean isLogItem(int itemId) {
        return logItemIds.contains(itemId);
    }

    public int blockCount() {
        return blocks.size();
    }

    public int itemCount() {
        return items.size();
    }
}
