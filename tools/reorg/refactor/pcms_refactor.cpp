// pcms-refactor: AST-based namespace moves for PCMS.
//
// Two passes over every translation unit of a configured build
// (compile_commands.json), each printing edits as JSON lines that
// apply_edits.py merges and applies:
//
//   --mode=qualify --moves=<file>                        (tree before the
//   moves)
//     <file> has one `<path> <target namespace>` line per file whose code
//     moves; any number of targets are moved in one pass. Renames the
//     top-level `namespace pcms` blocks of those files to their target and
//     writes every reference that a move affects fully qualified with a
//     leading `::pcms::` (a marker; PCMS code never spells `::pcms`):
//     references to the moved declarations, references from moved code
//     whose relative qualifier would resolve differently afterwards, and
//     global namespaces a new namespace would hide (`mfem::` inside
//     pcms::backend::mfem becomes `::mfem::`).
//
//   --mode=spell                                        (tree after the moves)
//     Rewrites every marked qualifier to the shortest spelling that still
//     names the same entity from where it is used (clang's
//     tooling::replaceNestedName plus a class-member and parameter check).
//     In a .cpp file, an internal name (declared outside include/, not in a
//     detail namespace) used from outside its namespace is spelled as it
//     was before the move (bare inside namespace pcms or under `using
//     namespace pcms;`, otherwise pcms::X), and the file gets a
//     using-declaration for it in a `namespace pcms { ... }` block after its
//     includes. Uses inside #if regions are spelled relatively instead.
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring/Lookup.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace clang;

static llvm::cl::OptionCategory Cat("pcms-refactor");
static llvm::cl::opt<std::string> Mode("mode",
                                       llvm::cl::desc("qualify | spell"),
                                       llvm::cl::cat(Cat));
static llvm::cl::opt<std::string> Moves(
  "moves",
  llvm::cl::desc("file of `<path> <target namespace>` lines: the files whose "
                 "code moves, and where"),
  llvm::cl::cat(Cat));
static llvm::cl::opt<std::string> Root("root",
                                       llvm::cl::desc("repository root"),
                                       llvm::cl::cat(Cat));

static const char* Marker = "::pcms::";
static const char* BareMark = "/*u*/";

namespace
{

std::map<std::string, std::string> Moved; // repository-relative path -> target
std::set<std::string> Targets;
// per target: namespace names that become visible to the code moving there
std::map<std::string, std::set<std::string>> NewlyVisibleInit;

std::string parentOf(const std::string& ns)
{
  return ns.substr(0, ns.rfind("::"));
}
std::string lastOf(const std::string& ns)
{
  return ns.substr(ns.rfind("::") + 2);
}
bool within(const std::string& ns, const std::string& outer)
{
  return ns == outer || StringRef(ns).starts_with(outer + "::");
}

std::string realPath(StringRef p)
{
  llvm::SmallString<256> out;
  if (llvm::sys::fs::real_path(p, out))
    return p.str();
  return std::string(out);
}

void emit(llvm::json::Object obj)
{
  llvm::outs() << llvm::json::Value(std::move(obj)) << "\n";
}

void note(const std::string& what, const std::string& where)
{
  emit(llvm::json::Object{{"note", what}, {"at", where}});
}

class Tool
{
public:
  Tool(ASTContext& ctx) : ctx_(ctx), sm_(ctx.getSourceManager()) {}

  // repository-relative path of the file a location is spelled in, or "" if
  // the location is not editable (macro body, outside the repository,
  // generated file)
  std::string fileOf(SourceLocation loc)
  {
    if (loc.isInvalid())
      return "";
    if (loc.isMacroID()) {
      if (!sm_.isMacroArgExpansion(loc))
        return "";
      loc = sm_.getSpellingLoc(loc);
    }
    FileID fid = sm_.getFileID(loc);
    auto it = files_.find(fid);
    if (it != files_.end())
      return it->second;
    std::string rel;
    if (auto entry = sm_.getFileEntryRefForID(fid)) {
      std::string p = realPath(entry->getName());
      std::string root = Root + "/";
      if (p.rfind(root, 0) == 0)
        rel = p.substr(root.size());
    }
    files_[fid] = rel;
    return rel;
  }

  std::string where(SourceLocation loc)
  {
    return sm_.getSpellingLoc(loc).printToString(sm_);
  }

  void edit(SourceLocation begin, SourceLocation end, StringRef text)
  {
    // [begin, end) as file offsets in the spelling file
    std::string f = fileOf(begin);
    if (f.empty())
      return;
    begin = sm_.getSpellingLoc(begin);
    end = sm_.getSpellingLoc(end);
    unsigned b = sm_.getFileOffset(begin), e = sm_.getFileOffset(end);
    emit(llvm::json::Object{{"file", f},
                            {"offset", int64_t(b)},
                            {"length", int64_t(e - b)},
                            {"text", text.str()}});
  }

  // end (exclusive) of the token at loc
  SourceLocation tokenEnd(SourceLocation loc)
  {
    return Lexer::getLocForEndOfToken(sm_.getSpellingLoc(loc), 0, sm_,
                                      ctx_.getLangOpts());
  }

  // target namespace of the file a location is in, "" if the file stays
  std::string targetOf(SourceLocation loc)
  {
    std::string f = fileOf(loc);
    auto it = f.empty() ? Moved.end() : Moved.find(f);
    return it == Moved.end() ? "" : it->second;
  }
  bool isMovedFile(SourceLocation loc) { return !targetOf(loc).empty(); }

  static const NamespaceDecl* outermost(const DeclContext* dc)
  {
    const NamespaceDecl* top = nullptr;
    for (; dc; dc = dc->getParent())
      if (auto* ns = dyn_cast<NamespaceDecl>(dc))
        top = ns;
    return top;
  }

  // the declaration a reference names, normalized to the primary template /
  // the enum for unscoped enumerators
  static const NamedDecl* primary(const NamedDecl* d)
  {
    if (!d)
      return nullptr;
    if (auto* sh = dyn_cast<UsingShadowDecl>(d))
      d = sh->getTargetDecl();
    if (auto* fd = dyn_cast<FunctionDecl>(d))
      if (auto* t = fd->getPrimaryTemplate())
        d = t;
    if (auto* sp = dyn_cast<ClassTemplateSpecializationDecl>(d))
      d = sp->getSpecializedTemplate();
    if (auto* td = dyn_cast<TemplateDecl>(d))
      if (td->getTemplatedDecl())
        d = td;
    if (auto* ec = dyn_cast<EnumConstantDecl>(d)) {
      auto* ed = cast<EnumDecl>(ec->getDeclContext());
      if (!ed->isScoped())
        d = ed;
    }
    return d;
  }

  // namespace-level declaration in pcms whose (first) declaration is in a
  // moved file
  // its target namespace, "" if it stays
  std::string movedTo(const NamedDecl* d)
  {
    d = primary(d);
    if (!d || !isa<NamespaceDecl>(d->getDeclContext()))
      return "";
    if (isa<NamespaceDecl>(d))
      return "";
    const NamespaceDecl* top = outermost(d->getDeclContext());
    if (!top || top->getName() != "pcms")
      return "";
    const Decl* first = d->getCanonicalDecl();
    if (auto* td = dyn_cast<TemplateDecl>(d))
      first = td->getCanonicalDecl();
    std::string to = targetOf(first->getLocation());
    // already there (the spec was applied before): nothing to do
    if (!to.empty() &&
        within(
          cast<NamespaceDecl>(d->getDeclContext())->getQualifiedNameAsString(),
          to))
      return "";
    return to;
  }
  bool isMoved(const NamedDecl* d) { return !movedTo(d).empty(); }

  // "::pcms::<to>::<rest of the declaration's namespace>::"
  std::string newPrefix(const NamedDecl* d)
  {
    d = primary(d);
    auto* ns = cast<NamespaceDecl>(d->getDeclContext());
    std::string q = ns->getQualifiedNameAsString(); // pcms or pcms::detail ...
    std::string rest = q.size() > 4 ? q.substr(4) : ""; // "" or "::detail"
    return "::" + movedTo(d) + rest + "::";
  }

  static bool namespacesOnly(NestedNameSpecifierLoc q)
  {
    for (; q; q = q.getPrefix()) {
      auto k = q.getNestedNameSpecifier()->getKind();
      if (k != NestedNameSpecifier::Namespace &&
          k != NestedNameSpecifier::NamespaceAlias &&
          k != NestedNameSpecifier::Global)
        return false;
    }
    return true;
  }

  static std::string firstComponent(NestedNameSpecifierLoc q)
  {
    NestedNameSpecifierLoc first = q;
    while (first.getPrefix())
      first = first.getPrefix();
    auto* nns = first.getNestedNameSpecifier();
    if (nns->getKind() == NestedNameSpecifier::Global)
      return "::";
    if (auto* ns = nns->getAsNamespace())
      return ns->getNameAsString();
    if (auto* al = nns->getAsNamespaceAlias())
      return al->getNameAsString();
    return "";
  }

  // -------------------------------------------------------------- qualify
  void qualifyRef(NestedNameSpecifierLoc q, SourceLocation nameLoc,
                  const NamedDecl* d, bool declarator = false)
  {
    if (!d || nameLoc.isInvalid())
      return;
    if (fileOf(nameLoc).empty()) {
      if (nameLoc.isMacroID() && isMoved(d))
        note("reference to moved " + d->getQualifiedNameAsString() +
               " in a macro body",
             where(nameLoc));
      return;
    }
    std::string to = movedTo(d);
    if (!to.empty()) {
      if (q) {
        if (!namespacesOnly(q))
          return; // qualified through a class: the class reference moves
        replaceQualifier(q, newPrefix(d));
      } else if (targetOf(nameLoc) != to && !declarator) {
        // /*u*/ records that the name was written unqualified
        edit(nameLoc, nameLoc, std::string(BareMark) + newPrefix(d));
      }
      return;
    }
    // non-moved entity referenced from moved code with a relative qualifier
    // whose head will also name something in the target namespace
    std::string from = targetOf(nameLoc);
    // moved code naming something that is already in its target namespace
    // (pcms::backend::omega_h::X in code moving to pcms::backend::omega_h):
    // respelled, i.e. shortened, after the move
    if (q && !from.empty() && !declarator && namespacesOnly(q) &&
        isa<NamespaceDecl>(d->getDeclContext()) &&
        within(
          cast<NamespaceDecl>(d->getDeclContext())->getQualifiedNameAsString(),
          from)) {
      replaceQualifier(
        q,
        "::" +
          cast<NamespaceDecl>(d->getDeclContext())->getQualifiedNameAsString() +
          "::");
      return;
    }
    if (q && !from.empty() && !declarator) {
      std::string head = firstComponent(q);
      if (head.empty() || head == "::" || head == "pcms")
        return;
      if (TargetNested[from].count(head) &&
          isa<NamespaceDecl>(d->getDeclContext())) {
        const NamespaceDecl* top = outermost(d->getDeclContext());
        if (top && top->getName() == "pcms" && namespacesOnly(q))
          replaceQualifier(q, "::" +
                                cast<NamespaceDecl>(d->getDeclContext())
                                  ->getQualifiedNameAsString() +
                                "::");
      }
    }
  }

  void replaceQualifier(NestedNameSpecifierLoc q, const std::string& text)
  {
    // q.getSourceRange(): from the first component to the final '::'
    SourceRange r = q.getSourceRange();
    edit(r.getBegin(), tokenEnd(r.getEnd()), text);
  }

  void renameNamespace(const NamespaceDecl* nd)
  {
    if (nd->getName() != "pcms" || !isa<TranslationUnitDecl>(nd->getParent()))
      return;
    std::string to = targetOf(nd->getLocation());
    if (to.empty())
      return;
    // `namespace pcms::a::b {`: the block's full name; one already in the
    // target (code written for the new namespace) stays as it is, while
    // `namespace pcms::detail` becomes <target>::detail
    std::string full = "pcms";
    for (const NamespaceDecl* n = nd;;) {
      const NamespaceDecl* inner = nullptr;
      for (auto* d : n->decls())
        if (auto* c = dyn_cast<NamespaceDecl>(d); c && c->isNested())
          inner = c;
      if (!inner)
        break;
      full += "::" + inner->getNameAsString();
      n = inner;
    }
    if (within(full, to))
      return;
    // a block of using-declarations (added by an earlier move) brings names
    // into pcms for this file; it stays in pcms
    bool onlyUsings = !nd->decls_empty();
    for (auto* d : nd->decls())
      onlyUsings &= isa<UsingDecl>(d) || isa<UsingShadowDecl>(d);
    if (onlyUsings)
      return;
    SourceLocation loc = nd->getLocation();
    edit(loc, tokenEnd(loc), to);
    // closing comment: "} // namespace pcms" or "} // namespace pcms::detail"
    SourceLocation rb = sm_.getSpellingLoc(nd->getRBraceLoc());
    if (rb.isInvalid())
      return;
    bool invalid = false;
    const char* p = sm_.getCharacterData(rb, &invalid);
    if (invalid)
      return;
    const char* eol = p;
    while (*eol && *eol != '\n')
      ++eol;
    std::string line(p, eol);
    auto pos = line.find("// namespace pcms");
    if (pos != std::string::npos) {
      size_t after = pos + std::string("// namespace pcms").size();
      if (after == line.size() || line[after] == ':' || line[after] == ' ') {
        SourceLocation b = rb.getLocWithOffset(pos + 13); // at "pcms"
        edit(b, b.getLocWithOffset(4), to);
      }
    }
  }

  // per target: namespaces nested in the blocks that move there
  std::map<std::string, std::set<std::string>> TargetNested;
  // namespace names that become visible to the moved code (children of the
  // target namespace and of its parents below pcms, and its own components)
  std::map<std::string, std::set<std::string>> NewlyVisible;

  static std::string enclosingNamespace(const DeclContext* dc)
  {
    for (; dc; dc = dc->getParent())
      if (auto* ns = dyn_cast<NamespaceDecl>(dc))
        if (!ns->isAnonymousNamespace())
          return ns->getQualifiedNameAsString();
    return "";
  }

  // `mfem::X` inside pcms::backend::mfem would find the new namespace: write
  // the global namespace with a leading ::
  void anchorGlobalHead(NestedNameSpecifierLoc head, const DeclContext* useDC)
  {
    auto* nns = head.getNestedNameSpecifier();
    const NamespaceDecl* ns = nns->getAsNamespace();
    if (!ns || !isa<TranslationUnitDecl>(ns->getParent()) ||
        ns->getName() == "pcms")
      return;
    std::string name = ns->getNameAsString();
    SourceLocation loc = head.getLocalBeginLoc();
    if (fileOf(loc).empty())
      return;
    std::string enc = enclosingNamespace(useDC);
    std::string from = targetOf(loc);
    bool movedCode = !from.empty() && within(enc, "pcms");
    bool collides = false;
    if (movedCode) {
      collides = NewlyVisible[from].count(name) > 0;
    } else {
      // code that stays inside a target's parent sees the new namespace too
      for (auto& t : Targets)
        if (name == lastOf(t) && parentOf(t) != "pcms" &&
            within(enc, parentOf(t)))
          collides = true;
    }
    if (collides)
      edit(loc, loc, "::");
  }

  // -------------------------------------------------------------- spell
  std::set<std::string> usings;

  bool isSource(const std::string& f)
  {
    return StringRef(f).ends_with(".cpp") || StringRef(f).ends_with(".cc") ||
           StringRef(f).ends_with(".cxx");
  }

  bool isPublic(const NamedDecl* d)
  {
    for (auto* r : d->redecls()) {
      std::string f = fileOf(r->getLocation());
      if (StringRef(f).starts_with("include/"))
        return true;
    }
    return false;
  }

  // does lookup of `name` from `dc` (semantic chain) find something that is
  // not one of `targets` before reaching a namespace? (class members,
  // inherited members, function parameters)
  bool hiddenByScope(const DeclContext* dc, DeclarationName name,
                     const llvm::SmallPtrSetImpl<const Decl*>& targets)
  {
    for (; dc && !dc->isFileContext(); dc = dc->getParent()) {
      if (auto* rd = dyn_cast<CXXRecordDecl>(dc)) {
        if (!rd->hasDefinition())
          continue;
        if (recordHas(rd->getDefinition(), name, targets, 0))
          return true;
      } else if (auto* fd = dyn_cast<FunctionDecl>(dc)) {
        for (auto* p : fd->parameters())
          if (p->getDeclName() == name)
            return true;
      }
    }
    return false;
  }

  bool recordHas(const CXXRecordDecl* rd, DeclarationName name,
                 const llvm::SmallPtrSetImpl<const Decl*>& targets, int depth)
  {
    if (depth > 16)
      return false;
    for (auto* r : rd->lookup(name))
      if (!targets.count(r->getCanonicalDecl()))
        return true;
    for (auto& base : rd->bases()) {
      if (auto* b = base.getType()->getAsCXXRecordDecl())
        if (b->hasDefinition() &&
            recordHas(b->getDefinition(), name, targets, depth + 1))
          return true;
    }
    return false;
  }

  // lookup of `name` from `dc` through the enclosing namespaces (and global
  // `using namespace` directives); true if anything is found
  bool foundInNamespaces(const DeclContext* dc, DeclarationName name,
                         SourceLocation use)
  {
    for (; dc; dc = dc->getParent()) {
      if (!dc->isFileContext())
        continue;
      for (auto* r : dc->lookup(name))
        if (sm_.isBeforeInTranslationUnit(sm_.getSpellingLoc(r->getLocation()),
                                          sm_.getSpellingLoc(use)))
          return true;
      for (auto* ud : dc->using_directives())
        if (!ud->getNominatedNamespace()->lookup(name).empty())
          return true;
    }
    return false;
  }

  bool inConditional(SourceLocation loc)
  {
    SourceLocation s = sm_.getSpellingLoc(loc);
    if (!sm_.isInMainFile(s))
      return false;
    unsigned off = sm_.getFileOffset(s);
    for (auto& [b, e] : Conditionals)
      if (b <= off && off < e)
        return true;
    return false;
  }

  std::vector<std::pair<unsigned, unsigned>> Conditionals; // main file
  unsigned InsertAfter = 0;                                // main file offset
  bool GlobalUsingPcms = false;

  void spellQualifier(NestedNameSpecifierLoc full, const DeclContext* useDC,
                      bool declarator)
  {
    // the marked namespace part: ::pcms::...:: (Global + namespaces)
    std::vector<NestedNameSpecifierLoc> chain;
    for (auto q = full; q; q = q.getPrefix())
      chain.insert(chain.begin(), q);
    if (chain.size() < 2 || chain[0].getNestedNameSpecifier()->getKind() !=
                              NestedNameSpecifier::Global)
      return;
    auto* pcmsNs = chain[1].getNestedNameSpecifier()->getAsNamespace();
    if (!pcmsNs || pcmsNs->getName() != "pcms")
      return;
    size_t last = 1;
    while (last + 1 < chain.size() &&
           chain[last + 1].getNestedNameSpecifier()->getAsNamespace())
      ++last;
    NestedNameSpecifierLoc nsPart = chain[last];
    auto* ns = nsPart.getNestedNameSpecifier()->getAsNamespace();
    SourceRange r = nsPart.getSourceRange();
    SourceLocation partEnd = tokenEnd(r.getEnd());
    std::string file = fileOf(r.getBegin());
    if (file.empty()) {
      note("marked qualifier in a macro body", where(r.getBegin()));
      return;
    }
    // the name after the namespace part
    auto tok = Lexer::findNextToken(sm_.getSpellingLoc(r.getEnd()), sm_,
                                    ctx_.getLangOpts());
    if (!tok || !tok->is(tok::raw_identifier) ||
        tok->getRawIdentifier() == "template") {
      note("marked qualifier not followed by a name", where(r.getBegin()));
      return;
    }
    std::string name = tok->getRawIdentifier().str();
    DeclarationName dn(&ctx_.Idents.get(name));
    llvm::SmallPtrSet<const Decl*, 4> targets;
    const NamedDecl* target = nullptr;
    for (auto* d : ns->lookup(dn)) {
      targets.insert(d->getCanonicalDecl());
      if (auto* sh = dyn_cast<UsingShadowDecl>(d))
        targets.insert(sh->getTargetDecl()->getCanonicalDecl());
      target = d;
    }
    if (!target) {
      note("marked name " + name + " not found in " +
             ns->getQualifiedNameAsString(),
           where(r.getBegin()));
      return;
    }
    std::string nsName = ns->getQualifiedNameAsString(); // pcms::localization
    std::string qname = "::" + nsName + "::" + name;
    bool insideNs = false;
    for (auto* dc = useDC; dc; dc = dc->getParent())
      if (auto* n = dyn_cast<NamespaceDecl>(dc))
        if (n->getCanonicalDecl() == ns->getCanonicalDecl())
          insideNs = true;

    // was the name written unqualified before the move?
    bool bare = false;
    SourceLocation editBegin = r.getBegin();
    {
      SourceLocation b = sm_.getSpellingLoc(r.getBegin());
      unsigned off = sm_.getFileOffset(b);
      size_t n = strlen(BareMark);
      if (off >= n) {
        bool invalid = false;
        const char* p = sm_.getCharacterData(b, &invalid);
        if (!invalid && StringRef(p - n, n) == BareMark) {
          bare = true;
          editBegin = b.getLocWithOffset(-int(n));
        }
      }
    }
    std::string spelled;
    bool internalRule = !declarator && isSource(file) && !insideNs &&
                        nsName != "pcms" &&
                        nsName.find("detail") == std::string::npos &&
                        !isPublic(target) && !inConditional(r.getBegin());
    if (internalRule) {
      bool inPcms = false;
      for (auto* dc = useDC; dc; dc = dc->getParent())
        if (auto* n = dyn_cast<NamespaceDecl>(dc))
          if (n->getName() == "pcms" &&
              isa<TranslationUnitDecl>(n->getParent()))
            inPcms = true;
      bool pcmsHasName = false;
      for (auto* d : pcmsNs->lookup(dn))
        if (!targets.count(d->getCanonicalDecl()))
          pcmsHasName = true;
      if (pcmsHasName) {
        internalRule = false; // a using-declaration would conflict
      } else {
        bool hidden = hiddenByScope(useDC, dn, targets) ||
                      foundInNamespaces(useDC, dn, r.getBegin());
        // keep the form it had: bare stays bare where that still resolves,
        // pcms::X stays pcms::X
        if (bare && (inPcms || GlobalUsingPcms) && !hidden)
          spelled = "";
        else
          spelled = "pcms::";
        usings.insert(nsName.substr(6) + "::" + name); // drop "pcms::"
      }
    }
    if (!internalRule) {
      const DeclContext* ctx = useDC;
      std::string s = tooling::replaceNestedName(
        full.getNestedNameSpecifier(), r.getBegin(), ctx, target, qname);
      // s ends with name; strip a leading "::" that is not needed
      if (StringRef(s).starts_with("::pcms::"))
        s = s.substr(2);
      std::string qual = s.substr(0, s.size() - name.size());
      if (qual.empty() && hiddenByScope(useDC, dn, targets)) {
        // the bare name is hidden: start from the full name; the loop below
        // keeps the shortest suffix that lookup resolves to the namespace
        qual = nsName + "::";
      }
      // replaceNestedName compares namespace names, not lookup: it keeps a
      // qualifier when a namespace of the same name exists in any enclosing
      // scope, and shortens to a qualifier that an enclosing namespace of the
      // same name hides (detail:: inside backend::omega_h::detail). Lookup
      // finds the innermost one, so try every suffix and keep the shortest
      // that resolves to the same entity.
      std::vector<const NamespaceDecl*> chain;
      for (const DeclContext* dc = ns; dc; dc = dc->getParent())
        if (auto* n = dyn_cast<NamespaceDecl>(dc))
          chain.insert(chain.begin(), n);
      auto suffix = [&](size_t m) {
        std::string cand;
        for (size_t i = chain.size() - m; i < chain.size(); ++i)
          cand += chain[i]->getNameAsString() + "::";
        return cand;
      };
      auto resolves = [&](size_t m) {
        bool ok;
        if (m == 0) {
          auto found = lookupFirst(useDC, dn);
          ok = !found.empty();
          for (auto* f : found)
            ok &= targets.count(f->getCanonicalDecl()) > 0;
        } else {
          const NamespaceDecl* head = chain[chain.size() - m];
          auto found = lookupFirst(
            useDC, DeclarationName(&ctx_.Idents.get(head->getName())), true);
          ok = !found.empty();
          for (auto* f : found) {
            auto* n = dyn_cast<NamespaceDecl>(f);
            if (auto* al = dyn_cast<NamespaceAliasDecl>(f))
              n = al->getNamespace();
            ok &= n && n->getCanonicalDecl() == head->getCanonicalDecl();
          }
        }
        return ok;
      };
      for (size_t m = 0; m <= chain.size(); ++m) {
        std::string cand = suffix(m);
        if (cand.size() > qual.size())
          break;
        if (cand == qual) {
          // the qualifier replaceNestedName chose: keep it if it resolves,
          // otherwise use the shortest longer suffix that does
          size_t k = m;
          while (k < chain.size() && !resolves(k))
            ++k;
          qual = suffix(k);
          break;
        }
        if (cand.size() < qual.size() && resolves(m)) {
          qual = cand;
          break;
        }
      }
      spelled = qual;
    }
    edit(editBegin, partEnd, spelled);
  }

  // what unqualified lookup of `name` from `dc` finds first: the
  // declarations of the innermost scope that has any (class members,
  // parameters, namespace members and names brought in by using-directives)
  std::vector<const NamedDecl*> lookupFirst(const DeclContext* dc,
                                            DeclarationName name,
                                            bool namespaceName = false)
  {
    for (; dc; dc = dc->getParent()) {
      if (namespaceName && !dc->isFileContext())
        continue; // only namespaces (and aliases) can name a qualifier here
      if (auto* rd = dyn_cast<CXXRecordDecl>(dc)) {
        llvm::SmallPtrSet<const Decl*, 1> none;
        if (rd->hasDefinition() &&
            recordHas(rd->getDefinition(), name, none, 0))
          return {rd}; // a member: not the entity
        continue;
      }
      if (auto* fd = dyn_cast<FunctionDecl>(dc)) {
        for (auto* p : fd->parameters())
          if (p->getDeclName() == name)
            return {p};
        continue;
      }
      if (!dc->isFileContext())
        continue;
      std::vector<const NamedDecl*> found;
      for (auto* r : dc->lookup(name))
        found.push_back(r);
      for (auto* ud : dc->using_directives())
        for (auto* r : ud->getNominatedNamespace()->lookup(name))
          found.push_back(r);
      if (!found.empty())
        return found;
    }
    return {};
  }

  ASTContext& ctx_;
  SourceManager& sm_;
  std::map<FileID, std::string> files_;
};

class Visitor : public RecursiveASTVisitor<Visitor>
{
public:
  explicit Visitor(Tool& t) : t_(t) {}
  bool shouldVisitTemplateInstantiations() const { return false; }
  bool shouldVisitImplicitCode() const { return false; }

  bool qualify() const { return Mode == "qualify"; }

  // declaration context stack (semantic)
  bool TraverseDecl(Decl* d)
  {
    if (!d)
      return true;
    bool pushed = false;
    if (auto* dc = dyn_cast<DeclContext>(d)) {
      dcs_.push_back(dc);
      pushed = true;
    }
    decls_.push_back(d);
    bool ok = RecursiveASTVisitor::TraverseDecl(d);
    decls_.pop_back();
    if (pushed)
      dcs_.pop_back();
    return ok;
  }

  const DeclContext* useDC() const
  {
    return dcs_.empty() ? nullptr : dcs_.back();
  }

  bool collect = false; // qualify: first pass only collects namespaces

  bool VisitNamespaceDecl(NamespaceDecl* nd)
  {
    if (!qualify())
      return true;
    if (collect) {
      if (auto* p = dyn_cast<NamespaceDecl>(nd->getParent())) {
        std::string pq = p->getQualifiedNameAsString();
        for (auto& t : Targets)
          if (pq != "pcms" && within(t, pq))
            t_.NewlyVisible[t].insert(nd->getNameAsString());
      }
      auto* parent = dyn_cast<NamespaceDecl>(nd->getParent());
      if (parent && parent->getName() == "pcms" &&
          isa<TranslationUnitDecl>(parent->getParent()) &&
          t_.isMovedFile(nd->getLocation()))
        t_.TargetNested[t_.targetOf(nd->getLocation())].insert(
          nd->getNameAsString());
      return true;
    }
    t_.renameNamespace(nd);
    return true;
  }

  // ------------------------------------------------------------ qualify
  bool VisitDeclRefExpr(DeclRefExpr* e)
  {
    if (collect || !qualify())
      return true;
    auto kind = e->getNameInfo().getName().getNameKind();
    if (kind != DeclarationName::Identifier)
      return true;
    // found through a using-declaration: that declaration is updated instead
    if (!e->getQualifierLoc() && isa<UsingShadowDecl>(e->getFoundDecl()))
      return true;
    t_.qualifyRef(e->getQualifierLoc(), e->getLocation(), e->getDecl());
    return true;
  }

  bool VisitOverloadExpr(OverloadExpr* e)
  {
    if (collect || !qualify() ||
        e->getName().getNameKind() != DeclarationName::Identifier)
      return true;
    const NamedDecl* any = nullptr;
    bool allMoved = true, noneMoved = true;
    for (auto* d : e->decls()) {
      if (!e->getQualifierLoc() && isa<UsingShadowDecl>(d))
        return true; // found through a using-declaration
      any = d;
      allMoved &= t_.isMoved(d->getUnderlyingDecl());
      noneMoved &= !t_.isMoved(d->getUnderlyingDecl());
    }
    if (!any)
      return true;
    if (auto* ule = dyn_cast<UnresolvedLookupExpr>(e))
      if (ule->requiresADL() && allMoved && !e->getQualifierLoc() &&
          !t_.isMovedFile(e->getNameLoc())) {
        note("unqualified dependent call of moved " + any->getNameAsString() +
               " (relies on ADL); qualified",
             t_.where(e->getNameLoc()));
      }
    // a qualified call of a name that stays (detail::f<T>(...) in moved code
    // whose target has its own detail namespace) is checked like any other
    // qualified reference
    if (allMoved || (noneMoved && e->getQualifierLoc()))
      t_.qualifyRef(e->getQualifierLoc(), e->getNameLoc(),
                    any->getUnderlyingDecl());
    return true;
  }

  const NamedDecl* typeDecl(TypeLoc tl, SourceLocation& nameLoc)
  {
    tl = tl.getUnqualifiedLoc();
    if (auto t = tl.getAs<TagTypeLoc>()) {
      nameLoc = t.getNameLoc();
      return t.getDecl();
    }
    if (auto t = tl.getAs<TypedefTypeLoc>()) {
      nameLoc = t.getNameLoc();
      return t.getTypedefNameDecl();
    }
    if (auto t = tl.getAs<TemplateSpecializationTypeLoc>()) {
      nameLoc = t.getTemplateNameLoc();
      return t.getTypePtr()->getTemplateName().getAsTemplateDecl();
    }
    if (auto t = tl.getAs<DeducedTemplateSpecializationTypeLoc>()) {
      nameLoc = t.getTemplateNameLoc();
      return t.getTypePtr()->getTemplateName().getAsTemplateDecl();
    }
    if (auto t = tl.getAs<UsingTypeLoc>()) {
      // named through a using-declaration: that declaration is updated
      return nullptr;
    }
    return nullptr;
  }

  bool VisitElaboratedTypeLoc(ElaboratedTypeLoc tl)
  {
    if (collect || !qualify())
      return true;
    SourceLocation nameLoc;
    const NamedDecl* d = typeDecl(tl.getNamedTypeLoc(), nameLoc);
    if (!d)
      return true;
    handled_.insert(nameLoc.getRawEncoding());
    t_.qualifyRef(tl.getQualifierLoc(), nameLoc, d);
    return true;
  }

  bool visitPlainType(TypeLoc tl)
  {
    if (collect || !qualify())
      return true;
    SourceLocation nameLoc;
    const NamedDecl* d = typeDecl(tl, nameLoc);
    if (!d || handled_.count(nameLoc.getRawEncoding()))
      return true;
    t_.qualifyRef(NestedNameSpecifierLoc(), nameLoc, d);
    return true;
  }
  bool VisitTagTypeLoc(TagTypeLoc tl) { return visitPlainType(tl); }
  bool VisitTypedefTypeLoc(TypedefTypeLoc tl) { return visitPlainType(tl); }
  bool VisitTemplateSpecializationTypeLoc(TemplateSpecializationTypeLoc tl)
  {
    return visitPlainType(tl);
  }
  bool VisitDeducedTemplateSpecializationTypeLoc(
    DeducedTemplateSpecializationTypeLoc tl)
  {
    return visitPlainType(tl);
  }

  bool VisitDeclaratorDecl(DeclaratorDecl* d)
  {
    if (collect || !qualify() || !d->getQualifierLoc())
      return true;
    if (t_.isMoved(d))
      t_.qualifyRef(d->getQualifierLoc(), d->getLocation(), d,
                    /*declarator=*/true);
    return true;
  }

  bool VisitUsingDecl(UsingDecl* ud)
  {
    if (collect || !qualify())
      return true;
    for (auto* sh : ud->shadows()) {
      t_.qualifyRef(ud->getQualifierLoc(), ud->getNameInfo().getLoc(),
                    sh->getTargetDecl());
      break;
    }
    return true;
  }

  // NNS components that name a moved type (pcms::GridPointSearch2D::Result)
  // in qualify mode; marked qualifiers in spell mode
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc q)
  {
    if (!q)
      return true;
    if (collect) {
    } else if (qualify()) {
      auto* nns = q.getNestedNameSpecifier();
      if (!q.getPrefix() && useDC())
        t_.anchorGlobalHead(q, useDC());
      if (nns->getKind() == NestedNameSpecifier::TypeSpec ||
          nns->getKind() == NestedNameSpecifier::TypeSpecWithTemplate) {
        SourceLocation nameLoc;
        const NamedDecl* d = typeDecl(q.getTypeLoc(), nameLoc);
        if (d) {
          handled_.insert(nameLoc.getRawEncoding());
          t_.qualifyRef(q.getPrefix(), nameLoc, d);
        }
      }
    } else if (seen_.insert(q.getBeginLoc().getRawEncoding()).second) {
      const DeclContext* ctx = useDC();
      bool declarator = false;
      if (!decls_.empty()) {
        Decl* top = decls_.back();
        NestedNameSpecifierLoc own;
        if (auto* dd = dyn_cast<DeclaratorDecl>(top))
          own = dd->getQualifierLoc();
        else if (auto* td = dyn_cast<TagDecl>(top))
          own = td->getQualifierLoc();
        if (own && own.getBeginLoc() == q.getBeginLoc()) {
          declarator = true;
          ctx = top->getLexicalDeclContext();
        }
      }
      if (ctx)
        t_.spellQualifier(q, ctx, declarator);
    }
    return RecursiveASTVisitor::TraverseNestedNameSpecifierLoc(q);
  }

private:
  Tool& t_;
  std::vector<const DeclContext*> dcs_;
  std::vector<Decl*> decls_;
  std::set<unsigned> handled_;
  std::set<unsigned> seen_;
};

class Callbacks : public PPCallbacks
{
public:
  explicit Callbacks(SourceManager& sm) : sm_(sm) {}
  void If(SourceLocation loc, SourceRange, ConditionValueKind) override
  {
    open(loc);
  }
  void Ifdef(SourceLocation loc, const Token&, const MacroDefinition&) override
  {
    open(loc);
  }
  void Ifndef(SourceLocation loc, const Token&, const MacroDefinition&) override
  {
    open(loc);
  }
  void Endif(SourceLocation loc, SourceLocation ifLoc) override
  {
    if (!sm_.isInMainFile(loc) || stack_.empty())
      return;
    regions.push_back({stack_.back(), sm_.getFileOffset(loc)});
    stack_.pop_back();
  }
  void InclusionDirective(SourceLocation hash, const Token&, StringRef, bool,
                          CharSourceRange range, OptionalFileEntryRef,
                          StringRef, StringRef, const Module*, bool,
                          SrcMgr::CharacteristicKind) override
  {
    if (!sm_.isInMainFile(hash) || !stack_.empty())
      return;
    // end of the directive's line
    bool invalid = false;
    const char* p = sm_.getCharacterData(range.getEnd(), &invalid);
    if (invalid)
      return;
    unsigned off = sm_.getFileOffset(range.getEnd());
    while (*p && *p != '\n') {
      ++p;
      ++off;
    }
    lastInclude = off + (*p == '\n' ? 1 : 0);
  }
  std::vector<std::pair<unsigned, unsigned>> regions;
  unsigned lastInclude = 0;

private:
  void open(SourceLocation loc)
  {
    if (sm_.isInMainFile(loc))
      stack_.push_back(sm_.getFileOffset(loc));
  }
  SourceManager& sm_;
  std::vector<unsigned> stack_;
};

class Consumer : public ASTConsumer
{
public:
  Consumer(Callbacks* cb) : cb_(cb) {}
  void HandleTranslationUnit(ASTContext& ctx) override
  {
    if (ctx.getDiagnostics().hasErrorOccurred()) {
      note("translation unit has errors; skipped",
           ctx.getSourceManager()
             .getFileEntryRefForID(ctx.getSourceManager().getMainFileID())
             ->getName()
             .str());
      emit(llvm::json::Object{{"error", "parse"}});
      return;
    }
    Tool t(ctx);
    t.NewlyVisible = NewlyVisibleInit;
    t.Conditionals = cb_->regions;
    t.InsertAfter = cb_->lastInclude;
    auto& sm = ctx.getSourceManager();
    {
      StringRef buf = sm.getBufferData(sm.getMainFileID());
      t.GlobalUsingPcms = buf.contains("\nusing namespace pcms;\n");
    }
    Visitor v(t);
    if (Mode == "qualify") {
      // first collect namespaces nested in the moved blocks
      v.collect = true;
      v.TraverseDecl(ctx.getTranslationUnitDecl());
      v.collect = false;
      v.TraverseDecl(ctx.getTranslationUnitDecl());
    } else {
      v.TraverseDecl(ctx.getTranslationUnitDecl());
      if (!t.usings.empty()) {
        std::string main =
          t.fileOf(sm.getLocForStartOfFile(sm.getMainFileID()));
        llvm::json::Array names;
        for (auto& n : t.usings)
          names.push_back(n);
        emit(llvm::json::Object{{"usings", std::move(names)},
                                {"file", main},
                                {"insert", int64_t(t.InsertAfter)}});
      }
    }
  }

private:
  Callbacks* cb_;
};

class Action : public ASTFrontendAction
{
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& ci,
                                                 StringRef) override
  {
    auto cb = std::make_unique<Callbacks>(ci.getSourceManager());
    Callbacks* raw = cb.get();
    ci.getPreprocessor().addPPCallbacks(std::move(cb));
    return std::make_unique<Consumer>(raw);
  }
};

} // namespace

int main(int argc, const char** argv)
{
  auto parser = tooling::CommonOptionsParser::create(argc, argv, Cat);
  if (!parser) {
    llvm::errs() << parser.takeError();
    return 1;
  }
  if (Mode != "qualify" && Mode != "spell") {
    llvm::errs() << "--mode must be qualify or spell\n";
    return 1;
  }
  Root = realPath(Root.empty() ? std::string(".") : std::string(Root));
  if (Mode == "qualify") {
    std::ifstream in(Moves);
    if (!in) {
      llvm::errs() << "cannot read --moves\n";
      return 1;
    }
    for (std::string line; std::getline(in, line);) {
      auto h = line.find('#');
      if (h != std::string::npos)
        line = line.substr(0, h);
      auto [path, target] = StringRef(line).trim().split(' ');
      target = target.trim();
      if (path.empty())
        continue;
      if (!target.starts_with("pcms::") || target.contains(' ')) {
        llvm::errs() << "--moves: bad line: " << line << "\n";
        return 1;
      }
      Moved[path.str()] = target.str();
      Targets.insert(target.str());
    }
    // what becomes visible to the code moving to t: its own components, and
    // the other new namespaces that are siblings of t or of t's parents
    // (below pcms; existing child namespaces are added while parsing)
    for (auto& t : Targets) {
      for (StringRef rest = StringRef(t).drop_front(6); !rest.empty();) {
        auto [head, tail] = rest.split("::");
        NewlyVisibleInit[t].insert(head.str());
        rest = tail;
      }
      for (auto& u : Targets)
        if (parentOf(u) != "pcms" && within(t, parentOf(u)))
          NewlyVisibleInit[t].insert(lastOf(u));
    }
  }
  tooling::ClangTool tool(parser->getCompilations(),
                          parser->getSourcePathList());
  tool.appendArgumentsAdjuster(tooling::getInsertArgumentAdjuster(
    "-Wno-everything", tooling::ArgumentInsertPosition::END));
  return tool.run(tooling::newFrontendActionFactory<Action>().get());
}
