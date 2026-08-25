import { NavLink, Route, Routes } from "react-router-dom";

import Ask from "./routes/Ask";
import Evaluation from "./routes/Evaluation";

export default function App() {
  return (
    <div className="shell">
      <header className="top">
        <h1>rag-eval-platform</h1>
        <nav>
          <NavLink to="/" end>
            ask
          </NavLink>
          <NavLink to="/eval">evaluation</NavLink>
          <a href="https://github.com/salahalomar/rag-eval-platform">source</a>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<Ask />} />
        <Route path="/eval" element={<Evaluation />} />
        <Route path="*" element={<p className="state">No such page.</p>} />
      </Routes>
    </div>
  );
}
