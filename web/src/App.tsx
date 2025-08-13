import "./App.css";
import MVPDemo from "./components/MVPDemo";
import "./components/MVPDemo.css";
import { DarkModeProvider } from "./contexts/DarkModeContext";
import { DarkModeToggle } from "./components/DarkModeToggle";

function App() {
  return (
    <DarkModeProvider>
      <div className="app">
        <DarkModeToggle />
        <MVPDemo />
      </div>
    </DarkModeProvider>
  );
}

export default App;
