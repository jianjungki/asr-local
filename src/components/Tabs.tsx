import { createSignal, For, JSXElement, Show } from "solid-js";
import "./Tabs.css";

type TabProps = {
  tabs: {
    label: string;
    content: JSXElement;
  }[];
};

function Tabs(props: TabProps) {
  const [activeIndex, setActiveIndex] = createSignal(0);

  return (
    <div class="tabs-container">
      <div class="tabs-list">
        <For each={props.tabs}>
          {(tab, index) => (
            <button
              class="tab-button"
              classList={{ active: activeIndex() === index() }}
              onClick={() => setActiveIndex(index())}
            >
              {tab.label}
            </button>
          )}
        </For>
      </div>
      <div class="tab-content">
        <For each={props.tabs}>
          {(tab, index) => (
            <Show when={activeIndex() === index()}>
              <div class="tab-panel">{tab.content}</div>
            </Show>
          )}
        </For>
      </div>
    </div>
  );
}

export default Tabs;