package com.mcai.agent;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.message.v1.ClientSendMessageEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.text.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Locale;

/**
 * Client entry point.
 *
 * Chat-swallow: a "." prefixed message typed in chat is treated as a client
 * command, handled locally, and NOT forwarded to the server. This mirrors
 * Meteor's sendChat intercept; here it uses Fabric's ClientSendMessageEvents
 * .ALLOW_CHAT, the supported 1.21 hook for cancelling an outgoing chat message.
 *
 * Tick-hook: the agent loop runs on ClientTickEvents.END_CLIENT_TICK, i.e. once
 * per client tick after vanilla tick logic, mirroring Meteor's TickEvent.Post.
 */
public final class McaiAgentClient implements ClientModInitializer {
    public static final Logger LOG = LoggerFactory.getLogger("mcai-agent");
    public static final String PREFIX = ".";

    // Absolute paths to the shared weights + schema (bundled at these locations
    // in the repo). Reading by absolute path avoids copying large weights into
    // resources for now.
    private static final Path WEIGHTS_DIR = Path.of("/home/user/github/mcai/weights");
    private static final Path REGISTRY_JSON = Path.of("/home/user/github/mcai/schema/registry.json");

    private SchemaRegistry registry;
    private AgentRunner runner; // non-null while running
    private String loadedName;

    @Override
    public void onInitializeClient() {
        // Swallow "." commands client-side: handle and cancel the send.
        ClientSendMessageEvents.ALLOW_CHAT.register(message -> {
            if (message.startsWith(PREFIX)) {
                handleCommand(message.substring(PREFIX.length()).trim());
                return false; // cancel: do NOT send to server
            }
            return true;
        });

        // Per-tick agent loop.
        ClientTickEvents.END_CLIENT_TICK.register(this::onClientTick);

        LOG.info("MCAI Agent initialized. Use .run <name>, .stop, .status");
    }

    private void onClientTick(MinecraftClient mc) {
        if (runner == null) {
            return;
        }
        try {
            runner.tick(mc);
        } catch (Exception e) {
            LOG.error("Agent tick failed; stopping.", e);
            stop(mc, "error: " + e.getMessage());
        }
    }

    private void handleCommand(String cmd) {
        MinecraftClient mc = MinecraftClient.getInstance();
        String[] parts = cmd.split("\\s+");
        if (parts.length == 0 || parts[0].isEmpty()) {
            return;
        }
        switch (parts[0].toLowerCase(Locale.ROOT)) {
            case "run" -> {
                if (parts.length < 2) {
                    feedback(mc, "usage: .run <name>");
                    return;
                }
                run(mc, parts[1]);
            }
            case "stop" -> stop(mc, "stopped");
            case "status" -> feedback(mc, runner != null
                ? "running: " + loadedName
                : "idle");
            default -> feedback(mc, "unknown command: " + parts[0]);
        }
    }

    private void run(MinecraftClient mc, String name) {
        try {
            if (registry == null) {
                registry = SchemaRegistry.load(REGISTRY_JSON);
                LOG.info("Loaded registry: {} blocks, {} items", registry.blockCount(), registry.itemCount());
            }
            if (runner != null) {
                runner.release(mc);
                runner.close();
                runner = null;
            }
            Path onnx = WEIGHTS_DIR.resolve(name + ".onnx");
            OnnxPolicy policy = new OnnxPolicy(onnx);
            runner = new AgentRunner(policy, registry);
            loadedName = name;
            feedback(mc, "running policy: " + name);
            LOG.info("Loaded policy {}", onnx);
        } catch (Exception e) {
            LOG.error("Failed to start policy {}", name, e);
            feedback(mc, "failed to load " + name + ": " + e.getMessage());
        }
    }

    private void stop(MinecraftClient mc, String why) {
        if (runner != null) {
            runner.release(mc);
            runner.close();
            runner = null;
            loadedName = null;
        }
        feedback(mc, why);
    }

    private void feedback(MinecraftClient mc, String msg) {
        if (mc.player != null) {
            mc.player.sendMessage(Text.literal("[mcai] " + msg), false);
        }
        LOG.info("[mcai] {}", msg);
    }
}
