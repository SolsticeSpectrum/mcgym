package com.mcgym;

import com.mcgym.agent.Runner;
import com.mcgym.policy.Policy;
import com.mcgym.schema.Registry;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.message.v1.ClientSendMessageEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.text.Text;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Locale;

public final class McGym implements ClientModInitializer {

    public static final Logger LOG = LoggerFactory.getLogger("mcgym");
    public static final String PREFIX = ".";

    private Registry registry;
    private Runner runner;
    private String loaded;

    @Override
    public void onInitializeClient() {
        // swallow "." commands client side, handle and cancel the send
        ClientSendMessageEvents.ALLOW_CHAT.register(message -> {
            if (message.startsWith(PREFIX)) {
                handle(message.substring(PREFIX.length()).trim());
                return false;
            }
            return true;
        });

        ClientTickEvents.END_CLIENT_TICK.register(this::tick);

        LOG.info("mcgym ready, use .run <name>, .stop, .status");
    }

    // repo root holding weights/ and schema/, no fallback
    private static Path home() {
        String home = System.getenv("MCGYM_HOME");
        if (home == null) throw new IllegalStateException("MCGYM_HOME not set, point it at the mcgym repo");
        return Path.of(home);
    }

    private void tick(MinecraftClient mc) {
        if (runner == null) return;
        try {
            runner.tick(mc);
        } catch (Exception e) {
            LOG.error("agent tick failed, stopping", e);
            stop(mc, "error: " + e.getMessage());
        }
    }

    private void handle(String cmd) {
        MinecraftClient mc = MinecraftClient.getInstance();
        String[] parts = cmd.split("\\s+");
        if (parts.length == 0 || parts[0].isEmpty()) return;

        switch (parts[0].toLowerCase(Locale.ROOT)) {
            case "run" -> {
                if (parts.length < 2) {
                    feedback(mc, "usage: .run <name>");
                    return;
                }
                run(mc, parts[1]);
            }
            case "stop" -> stop(mc, "stopped");
            case "status" -> feedback(mc, runner != null ? "running: " + loaded : "idle");
            default -> feedback(mc, "unknown command: " + parts[0]);
        }
    }

    private void run(MinecraftClient mc, String name) {
        try {
            Path home = home();
            if (registry == null) {
                registry = Registry.load(home.resolve("schema/registry.json"));
                LOG.info("registry loaded, {} blocks {} items", registry.blocks(), registry.items());
            }
            if (runner != null) {
                runner.release(mc);
                runner.close();
                runner = null;
            }
            Path onnx = home.resolve("weights").resolve(name + ".onnx");
            runner = new Runner(new Policy(onnx), registry);
            loaded = name;
            feedback(mc, "running policy: " + name);
        } catch (Exception e) {
            LOG.error("failed to start policy {}", name, e);
            feedback(mc, "failed to load " + name + ": " + e.getMessage());
        }
    }

    private void stop(MinecraftClient mc, String why) {
        if (runner != null) {
            runner.release(mc);
            runner.close();
            runner = null;
            loaded = null;
        }
        feedback(mc, why);
    }

    private void feedback(MinecraftClient mc, String msg) {
        if (mc.player != null) mc.player.sendMessage(Text.literal("[mcgym] " + msg), false);
        LOG.info("[mcgym] {}", msg);
    }
}
