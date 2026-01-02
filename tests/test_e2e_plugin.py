"""End-to-end tests for plugin loading and registration.

These tests verify that the plugin can be correctly loaded via setuptools
entry points, which is how llm discovers and loads plugins.
"""

import importlib
import sys

import pytest


class TestModuleImport:
    """Tests for basic module importability."""

    def test_module_can_be_imported(self):
        """Test that llm_claude_code module can be imported."""
        # This catches the ModuleNotFoundError early
        import llm_claude_code

        assert llm_claude_code is not None

    def test_module_has_required_exports(self):
        """Test that module exports all required symbols."""
        import llm_claude_code

        # Required exports for the plugin to work
        assert hasattr(llm_claude_code, "ClaudeCode")
        assert hasattr(llm_claude_code, "ClaudeCodeOptions")
        assert hasattr(llm_claude_code, "register_models")
        assert hasattr(llm_claude_code, "CLAUDE_MODELS")

    def test_module_reload(self):
        """Test that module can be reloaded without errors."""
        import llm_claude_code

        # Reload to ensure module is properly structured
        reloaded = importlib.reload(llm_claude_code)
        assert reloaded is not None
        assert hasattr(reloaded, "register_models")


class TestEntryPointLoading:
    """Tests for setuptools entry point loading."""

    def test_entry_point_exists(self):
        """Test that the llm entry point is defined."""
        from importlib.metadata import entry_points

        # Get all llm entry points
        eps = entry_points(group="llm")

        # Find our entry point
        claude_code_eps = [ep for ep in eps if ep.name == "claude-code"]
        assert len(claude_code_eps) == 1, "Expected exactly one claude-code entry point"

    def test_entry_point_value(self):
        """Test that entry point points to correct module."""
        from importlib.metadata import entry_points

        eps = entry_points(group="llm")
        claude_code_ep = next(ep for ep in eps if ep.name == "claude-code")

        # The entry point should point to our module
        assert claude_code_ep.value == "llm_claude_code"

    def test_entry_point_can_be_loaded(self):
        """Test that entry point can be loaded successfully.

        This is the exact operation that causes the ModuleNotFoundError
        if the package is not properly installed.
        """
        from importlib.metadata import entry_points

        eps = entry_points(group="llm")
        claude_code_ep = next(ep for ep in eps if ep.name == "claude-code")

        # This is what pluggy does internally - load the entry point
        module = claude_code_ep.load()

        assert module is not None
        # When loading a module (not a function), we get the module itself
        assert hasattr(module, "register_models")

    def test_entry_point_register_models_is_hookimpl(self):
        """Test that register_models is properly decorated as a hook implementation."""
        from importlib.metadata import entry_points

        eps = entry_points(group="llm")
        claude_code_ep = next(ep for ep in eps if ep.name == "claude-code")
        module = claude_code_ep.load()

        register_func = module.register_models

        # llm uses pluggy, which marks hookimpls with 'llm_impl' attribute
        assert hasattr(register_func, "llm_impl")


class TestLLMPluginDiscovery:
    """Tests for llm plugin system integration."""

    def test_llm_can_discover_plugin(self):
        """Test that llm's plugin manager can find and load the plugin."""
        import llm

        # Get all registered models
        models = llm.get_models()
        model_ids = [m.model_id for m in models]

        # Our models should be registered
        assert "claude-code" in model_ids

    def test_llm_can_get_model_by_id(self):
        """Test that llm.get_model works for our models."""
        import llm

        model = llm.get_model("claude-code")
        assert model is not None
        assert model.model_id == "claude-code"

    def test_llm_can_get_model_by_alias(self):
        """Test that llm can find models by their aliases."""
        import llm

        # claude-code has alias 'cc'
        model = llm.get_model("cc")
        assert model is not None
        assert model.model_id == "claude-code"

    def test_all_model_variants_registered(self):
        """Test that all model variants are properly registered."""
        import llm

        expected_models = [
            "claude-code",
            "claude-code-opus",
            "claude-code-sonnet",
            "claude-code-haiku",
        ]

        for model_id in expected_models:
            model = llm.get_model(model_id)
            assert model is not None, f"Model {model_id} not found"
            assert model.model_id == model_id


class TestPackageMetadata:
    """Tests for package metadata correctness."""

    def test_package_metadata_exists(self):
        """Test that package metadata is accessible."""
        from importlib.metadata import metadata

        meta = metadata("llm-claude-code")
        assert meta is not None

    def test_package_version(self):
        """Test that package version is defined."""
        from importlib.metadata import version

        pkg_version = version("llm-claude-code")
        assert pkg_version is not None
        assert len(pkg_version) > 0

    def test_package_dependencies(self):
        """Test that required dependencies are declared."""
        from importlib.metadata import requires

        deps = requires("llm-claude-code")
        assert deps is not None

        # Check llm is in dependencies
        dep_names = [d.split()[0].split(">")[0].split("<")[0] for d in deps]
        assert "llm" in dep_names
